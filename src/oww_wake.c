// openWakeWord runtime harness (TFLite C API)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <sys/resource.h>

#include "tensorflow/lite/c/c_api.h"
#include <tinyalsa/pcm.h>

#define SR            16000
#define STEP          1280
#define MEL_LOOKBACK  (STEP + 160 * 3)
#define MEL_BINS      32
#define MEL_WIN       76
#define MEL_CAP       (10 * 97)
#define EMB_DIM       96
#define FEAT_WIN      16
#define FEAT_CAP      120 
#define MEL_IO_MAX    (MEL_LOOKBACK / 160 * MEL_BINS)

//  stop command capture on end of speech
#define EP_K_SPEECH   2.5 
#define EP_K_SIL      1.5 
#define EP_FLOOR_MIN  0.00008 
#define EP_FLOOR_A    0.97

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int sig) { (void)sig; g_stop = 1; }

static double now_s(void) {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec * 1e-9;
}

static long peak_rss_kb(void) {
	struct rusage r;
	getrusage(RUSAGE_SELF, &r);
	return r.ru_maxrss;
}

// chain: shared mel -> embedding -> classifier streaming state.
typedef struct {
	TfLiteModel *mm, *em, *wm;
	TfLiteInterpreter *mel, *emb, *wake;

	float raw[MEL_LOOKBACK];
	long  raw_len;
	float mel_buf[MEL_CAP][MEL_BINS];
	long  mel_n;
	float feat_buf[FEAT_CAP][EMB_DIM];
	long  feat_n;
	int   cur_mel_len;

	float mel_io[MEL_IO_MAX];
	float emb_in[MEL_WIN * MEL_BINS];
	float feat_out[EMB_DIM];
	float cls_in[FEAT_WIN * EMB_DIM];

	double t_mel, t_emb, t_cls;
	int    n_mel, n_emb, n_cls;
} Chain;

static TfLiteInterpreter *make_interp(const char *path, int threads, TfLiteModel **keep) {
	TfLiteModel *m = TfLiteModelCreateFromFile(path);
	if (!m) { fprintf(stderr, "load model %s failed\n", path); return NULL; }
	TfLiteInterpreterOptions *o = TfLiteInterpreterOptionsCreate();
	TfLiteInterpreterOptionsSetNumThreads(o, threads);
	TfLiteInterpreter *it = TfLiteInterpreterCreate(m, o);
	TfLiteInterpreterOptionsDelete(o);
	if (!it) { fprintf(stderr, "create interp %s failed\n", path); return NULL; }
	*keep = m;
	return it;
}

// reset streaming buffers after a detection
static void chain_reset(Chain *c) {
	c->raw_len = 0;
	for (int i = 0; i < MEL_WIN; i++)
		for (int j = 0; j < MEL_BINS; j++) c->mel_buf[i][j] = 1.0f;
	c->mel_n = MEL_WIN;
	c->feat_n = 0;
	c->cur_mel_len = -1;
}

static int chain_init(Chain *c, const char *mel_p, const char *emb_p,
                      const char *wake_p, int threads) {
	memset(c, 0, sizeof(*c));
	c->mel = make_interp(mel_p, threads, &c->mm);
	c->emb = make_interp(emb_p, threads, &c->em);
	c->wake = make_interp(wake_p, threads, &c->wm);
	if (!c->mel || !c->emb || !c->wake) return -1;
	if (TfLiteInterpreterAllocateTensors(c->emb) != kTfLiteOk ||
	    TfLiteInterpreterAllocateTensors(c->wake) != kTfLiteOk) {
		fprintf(stderr, "allocate emb/wake failed\n");
		return -1;
	}
	chain_reset(c);
	return 0;
}

static void chain_free(Chain *c) {
	TfLiteInterpreterDelete(c->mel);
	TfLiteInterpreterDelete(c->emb);
	TfLiteInterpreterDelete(c->wake);
	TfLiteModelDelete(c->mm);
	TfLiteModelDelete(c->em);
	TfLiteModelDelete(c->wm);
}

// push one 1280-sample chunk; returns the wake score
static float chain_push(Chain *c, const int16_t *chunk) {
	// 1) slide the <=1760-sample raw window
	long over = c->raw_len + STEP - MEL_LOOKBACK;
	if (over < 0) over = 0;
	if (over > 0) {
		memmove(c->raw, c->raw + over, (c->raw_len - over) * sizeof(float));
		c->raw_len -= over;
	}
	for (int i = 0; i < STEP; i++) c->raw[c->raw_len + i] = (float)chunk[i];
	c->raw_len += STEP;

	// 2) melspectrogram over the raw window
	if ((int)c->raw_len != c->cur_mel_len) {
		int dims[2] = {1, (int)c->raw_len};
		TfLiteInterpreterResizeInputTensor(c->mel, 0, dims, 2);
		if (TfLiteInterpreterAllocateTensors(c->mel) != kTfLiteOk) {
			fprintf(stderr, "allocate mel failed\n");
			return -1;
		}
		c->cur_mel_len = (int)c->raw_len;
	}
	TfLiteTensor *mi = TfLiteInterpreterGetInputTensor(c->mel, 0);
	TfLiteTensorCopyFromBuffer(mi, c->raw, c->raw_len * sizeof(float));
	double a = now_s();
	if (TfLiteInterpreterInvoke(c->mel) != kTfLiteOk) { fprintf(stderr, "mel invoke failed\n"); return -1; }
	c->t_mel += now_s() - a; c->n_mel++;
	const TfLiteTensor *mo = TfLiteInterpreterGetOutputTensor(c->mel, 0);
	long mo_floats = TfLiteTensorByteSize(mo) / sizeof(float);
	if (mo_floats > MEL_IO_MAX) {
		/* a swapped/wrong mel model would overflow mel_io */
		fprintf(stderr, "mel output %ld floats > buffer %d; wrong model?\n",
		        mo_floats, MEL_IO_MAX);
		return -1;
	}
	long frames = mo_floats / MEL_BINS;
	TfLiteTensorCopyToBuffer(mo, c->mel_io, mo_floats * sizeof(float));

	// 3) append transformed frames to mel_buf
	for (long fr = 0; fr < frames; fr++) {
		if (c->mel_n >= MEL_CAP) {
			memmove(c->mel_buf, c->mel_buf[1], (MEL_CAP - 1) * MEL_BINS * sizeof(float));
			c->mel_n = MEL_CAP - 1;
		}
		for (int j = 0; j < MEL_BINS; j++)
			c->mel_buf[c->mel_n][j] = c->mel_io[fr * MEL_BINS + j] / 10.0f + 2.0f;
		c->mel_n++;
	}

	// 4) embedding from the last 76 mel frames
	if (c->mel_n >= MEL_WIN) {
		for (int fr = 0; fr < MEL_WIN; fr++)
			memcpy(c->emb_in + fr * MEL_BINS, c->mel_buf[c->mel_n - MEL_WIN + fr], MEL_BINS * sizeof(float));
		TfLiteTensor *ei = TfLiteInterpreterGetInputTensor(c->emb, 0);
		TfLiteTensorCopyFromBuffer(ei, c->emb_in, sizeof(c->emb_in));
		a = now_s();
		if (TfLiteInterpreterInvoke(c->emb) != kTfLiteOk) { fprintf(stderr, "emb invoke failed\n"); return -1; }
		c->t_emb += now_s() - a; c->n_emb++;
		const TfLiteTensor *eo = TfLiteInterpreterGetOutputTensor(c->emb, 0);
		TfLiteTensorCopyToBuffer(eo, c->feat_out, sizeof(c->feat_out));

		if (c->feat_n >= FEAT_CAP) {
			memmove(c->feat_buf, c->feat_buf[1], (FEAT_CAP - 1) * EMB_DIM * sizeof(float));
			c->feat_n = FEAT_CAP - 1;
		}
		memcpy(c->feat_buf[c->feat_n++], c->feat_out, sizeof(c->feat_out));
	}

	// 5) classifier on the last 16 features
	if (c->feat_n < FEAT_WIN) return -1.0f;
	for (int fr = 0; fr < FEAT_WIN; fr++)
		memcpy(c->cls_in + fr * EMB_DIM, c->feat_buf[c->feat_n - FEAT_WIN + fr], EMB_DIM * sizeof(float));
	TfLiteTensor *ci = TfLiteInterpreterGetInputTensor(c->wake, 0);
	TfLiteTensorCopyFromBuffer(ci, c->cls_in, sizeof(c->cls_in));
	a = now_s();
	if (TfLiteInterpreterInvoke(c->wake) != kTfLiteOk) { fprintf(stderr, "wake invoke failed\n"); return -1; }
	c->t_cls += now_s() - a; c->n_cls++;
	const TfLiteTensor *co = TfLiteInterpreterGetOutputTensor(c->wake, 0);
	float score = 0;
	TfLiteTensorCopyToBuffer(co, &score, sizeof(float));
	return score;
}

static int16_t *load_wav(const char *path, long *n_out) {
	FILE *f = fopen(path, "rb");
	if (!f) { fprintf(stderr, "open %s failed\n", path); return NULL; }
	char id[4]; uint32_t sz;
	if (fread(id, 1, 4, f) != 4 || memcmp(id, "RIFF", 4)) goto bad;
	fseek(f, 4, SEEK_CUR);
	if (fread(id, 1, 4, f) != 4 || memcmp(id, "WAVE", 4)) goto bad;

	uint16_t ch = 1, bits = 16; uint32_t rate = SR;
	int16_t *data = NULL; long nsamp = 0;
	while (fread(id, 1, 4, f) == 4 && fread(&sz, 4, 1, f) == 1) {
		if (!memcmp(id, "fmt ", 4)) {
			uint8_t fmt[16]; long want = sz < 16 ? sz : 16;
			if (fread(fmt, 1, want, f) != (size_t)want) goto bad;
			ch   = fmt[2] | (fmt[3] << 8);
			rate = fmt[4] | (fmt[5] << 8) | (fmt[6] << 16) | ((uint32_t)fmt[7] << 24);
			bits = fmt[14] | (fmt[15] << 8);
			if (sz > 16) fseek(f, sz - 16, SEEK_CUR);
		} else if (!memcmp(id, "data", 4)) {
			long bytes = sz;
			uint8_t *raw = malloc(bytes);
			if (fread(raw, 1, bytes, f) != (size_t)bytes) { free(raw); goto bad; }
			long frames = bytes / (bits / 8) / (ch ? ch : 1);
			data = malloc(frames * sizeof(int16_t));
			const int16_t *s16 = (const int16_t *)raw;
			for (long i = 0; i < frames; i++) {
				int32_t acc = 0;
				for (int cc = 0; cc < ch; cc++) acc += s16[i * ch + cc];
				data[i] = (int16_t)(acc / ch);
			}
			nsamp = frames;
			free(raw);
		} else {
			fseek(f, sz + (sz & 1), SEEK_CUR);
		}
	}
	fclose(f);
	if (!data) { fprintf(stderr, "%s: no data chunk\n", path); return NULL; }
	if (rate != SR || bits != 16)
		fprintf(stderr, "warn %s: rate=%u bits=%u (expected 16k/16)\n", path, rate, bits);
	*n_out = nsamp;
	return data;
bad:
	fclose(f);
	fprintf(stderr, "%s: bad WAV\n", path);
	return NULL;
}

static void put_u32(FILE *f, uint32_t v) { fputc(v, f); fputc(v >> 8, f); fputc(v >> 16, f); fputc(v >> 24, f); }
static void put_u16(FILE *f, uint16_t v) { fputc(v, f); fputc(v >> 8, f); }

static int write_wav_s16(const char *path, const int16_t *data, long n) {
	FILE *f = fopen(path, "wb");
	if (!f) { fprintf(stderr, "open %s for write failed\n", path); return -1; }
	uint32_t bytes = n * 2;
	fwrite("RIFF", 1, 4, f); put_u32(f, 36 + bytes); fwrite("WAVE", 1, 4, f);
	fwrite("fmt ", 1, 4, f); put_u32(f, 16); put_u16(f, 1); put_u16(f, 1);
	put_u32(f, SR); put_u32(f, SR * 2); put_u16(f, 2); put_u16(f, 16);
	fwrite("data", 1, 4, f); put_u32(f, bytes);
	size_t wr = fwrite(data, 2, n, f);
	int close_err = fclose(f); /* flush catches buffered short writes (ENOSPC) */
	if (wr != (size_t)n || close_err != 0) {
		fprintf(stderr, "%s: short write (disk full?), dropping partial clip\n", path);
		remove(path);
		return -1;
	}
	return 0;
}

// file mode
static int run_file(Chain *c, const char *wav_p, double pad_sec, double threshold, int verbose) {
	long clip_n = 0;
	int16_t *clip = load_wav(wav_p, &clip_n);
	if (!clip) return 1;
	long pad_n = (long)(pad_sec * SR);
	long total_n = pad_n + clip_n;
	int16_t *audio = calloc(total_n, sizeof(int16_t));
	memcpy(audio + pad_n, clip, clip_n * sizeof(int16_t));
	free(clip);
	printf("clip: %.2fs + %.2fs lead-in = %ld samples (%ld steps)\n",
	       clip_n / (double)SR, pad_sec, total_n, total_n / STEP);

	double peak = 0, t0 = now_s();
	for (long off = 0; off + STEP <= total_n; off += STEP) {
		float s = chain_push(c, audio + off);
		if (s >= 0 && s > peak) peak = s;
		if (verbose && s > 0.01) printf("  step %4ld  score=%.4f\n", off / STEP, s);
	}
	double wall = now_s() - t0;
	free(audio);

	printf("\n--- result ---\n");
	printf("peak score : %.4f   %s (threshold %.2f)\n", peak,
	       peak >= threshold ? "FIRE" : "no-fire", threshold);
	printf("per-step   : mel %.2f ms (n=%d), emb %.2f ms (n=%d), cls %.3f ms (n=%d)\n",
	       c->n_mel ? c->t_mel / c->n_mel * 1e3 : 0, c->n_mel,
	       c->n_emb ? c->t_emb / c->n_emb * 1e3 : 0, c->n_emb,
	       c->n_cls ? c->t_cls / c->n_cls * 1e3 : 0, c->n_cls);
	printf("wall       : %.2f s for %.2f s audio\n", wall, total_n / (double)SR);
	printf("peak RSS   : %ld KiB (%.1f MB)\n", peak_rss_kb(), peak_rss_kb() / 1024.0);
	return 0;
}

// mic mode
static inline int16_t gained_s16(int32_t s, double gain) {
	double v = (double)s / 65536.0 * gain;
	if (v > 32767.0) v = 32767.0;
	if (v < -32768.0) v = -32768.0;
	return (int16_t)lrint(v);
}

static int write_cmd_clip(const char *path, const int32_t *cmd, long n) {
	double pk = 1e-9;
	for (long i = 0; i < n; i++) {
		double a = fabs((double)cmd[i] / 2147483648.0);
		if (a > pk) pk = a;
	}
	double scale = 0.5 / pk;
	int16_t *out = malloc(n * sizeof(int16_t));
	for (long i = 0; i < n; i++) {
		double v = (double)cmd[i] / 2147483648.0 * scale * 32767.0;
		if (v > 32767.0) v = 32767.0;
		if (v < -32768.0) v = -32768.0;
		out[i] = (int16_t)lrint(v);
	}
	int rc = write_wav_s16(path, out, n);
	free(out);
	return rc;
}

static double step_rms_s32(const int32_t *s, int n) {
	double acc = 0;
	for (int i = 0; i < n; i++) { double v = (double)s[i] / 2147483648.0; acc += v * v; }
	return sqrt(acc / n);
}

static int run_mic(Chain *c, const char *dev, double threshold, double gain,
                   double cmd_secs, double cooldown, const char *clip_path,
                   int endpoint_on, double min_secs, double hang_secs) {
	unsigned int card = 0, device = 0;
	sscanf(dev, "hw:%u,%u", &card, &device);

	struct pcm_config cfg = {
		.channels = 1,
		.rate = SR,
		.period_size = 1024,
		.period_count = 4,
		.format = PCM_FORMAT_S32_LE,
	};
	struct pcm *pcm = pcm_open(card, device, PCM_IN, &cfg);
	if (!pcm || !pcm_is_ready(pcm)) {
		fprintf(stderr, "pcm_open hw:%u,%u failed: %s\n", card, device,
		        pcm ? pcm_get_error(pcm) : "(null)");
		if (pcm) pcm_close(pcm);
		return 1;
	}

	signal(SIGINT, on_signal);
	signal(SIGTERM, on_signal);

	long cmd_max    = (long)(cmd_secs * SR);
	long min_n      = (long)(min_secs * SR);
	long hang_steps = (long)(hang_secs * SR / STEP);
	if (hang_steps < 1) hang_steps = 1;
	int32_t *cmd = malloc(cmd_max * sizeof(int32_t));
	long cmd_n = 0;
	enum { LISTENING, RECORDING } state = LISTENING;
	double cooldown_until = 0;

	double floor_rms = 1.0;
	double rec_floor = EP_FLOOR_MIN;
	int speech_started = 0;
	long sil_steps = 0;
	double rec_maxrms = 0, rec_minrms = 1e9;

	int32_t stage[STEP];
	long stage_n = 0;
	int32_t rd[512];
	int errs = 0;

	printf("MIC listening dev=hw:%u,%u thr=%.2f gain=%.1f cmd<=%.1fs endpoint=%d min=%.1fs hang=%.1fs cooldown=%.1fs clip=%s\n",
	       card, device, threshold, gain, cmd_secs, endpoint_on, min_secs, hang_secs, cooldown, clip_path);

	while (!g_stop) {
		int got = pcm_readi(pcm, rd, 512);
		if (got <= 0) {
			if (++errs > 50) { fprintf(stderr, "too many read errors, abort\n"); break; }
			continue;
		}
		errs = 0;
		for (int i = 0; i < got; i++) {
			stage[stage_n++] = rd[i];
			if (stage_n < STEP) continue;
			stage_n = 0;
			double rms = step_rms_s32(stage, STEP);

			if (state == LISTENING) {
				// adaptive noise floor
				if (rms < floor_rms) floor_rms += (rms - floor_rms) * 0.5;
				else                 floor_rms += (rms - floor_rms) * 0.01;
				if (floor_rms < EP_FLOOR_MIN) floor_rms = EP_FLOOR_MIN;
				int16_t chunk16[STEP];
				for (int k = 0; k < STEP; k++) chunk16[k] = gained_s16(stage[k], gain);
				float score = chain_push(c, chunk16);
				if (score >= threshold && now_s() >= cooldown_until) {
					printf("WAKE %.3f\n", score);
					state = RECORDING;
					cmd_n = 0;
					rec_floor = floor_rms > EP_FLOOR_MIN ? floor_rms : EP_FLOOR_MIN;
					speech_started = 0;
					sil_steps = 0;
					rec_maxrms = 0;
					rec_minrms = 1e9;
				}
			} else { // recording
				for (int k = 0; k < STEP && cmd_n < cmd_max; k++) cmd[cmd_n++] = stage[k];
				if (rms > rec_maxrms) rec_maxrms = rms;
				if (rms < rec_minrms) rec_minrms = rms;
				int stop = 0;
				if (endpoint_on) {
					if (rms > rec_floor * EP_K_SPEECH) { speech_started = 1; sil_steps = 0; }
					else if (rms < rec_floor * EP_K_SIL) { sil_steps++; }
					if (speech_started && cmd_n >= min_n && sil_steps >= hang_steps) stop = 1;
				}
				if (stop || cmd_n >= cmd_max) {
					if (write_cmd_clip(clip_path, cmd, cmd_n) == 0)
						printf("CLIP %s\n", clip_path);
					printf("CMDLEN %.2f %s floor=%.5f rms[min=%.5f max=%.5f] kspeech=%.5f ksil=%.5f speech=%d sil=%ld\n",
					       cmd_n / (double)SR, stop ? "endpoint" : "maxcap", rec_floor,
					       rec_minrms, rec_maxrms, rec_floor * EP_K_SPEECH, rec_floor * EP_K_SIL,
					       speech_started, (long)sil_steps);
					chain_reset(c);
					state = LISTENING;
					cooldown_until = now_s() + cooldown;
				}
			}
		}
	}

	free(cmd);
	pcm_close(pcm);
	printf("MIC stopped\n");
	return 0;
}

int main(int argc, char **argv) {
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (argc < 4) {
		fprintf(stderr,
		        "usage:\n"
		        "  file: %s mel emb wake clip.wav [--pad-sec S] [--threads N] [--threshold T] [--verbose]\n"
		        "  mic : %s mel emb wake --mic hw:0,0 [--threshold T] [--threads N] [--gain G]\n"
		        "        [--cmd-secs MAX] [--min-secs S] [--hang-secs S] [--no-endpoint]\n"
		        "        [--cooldown S] [--clip /path.wav]\n",
		        argv[0], argv[0]);
		return 2;
	}
	const char *mel_p = argv[1], *emb_p = argv[2], *wake_p = argv[3];
	const char *wav_p = NULL, *mic_dev = NULL, *clip_path = "/tmp/voice_cmd.wav";
	double pad_sec = 2.0, threshold = 0.5, gain = 16.0, cmd_secs = 5.1, cooldown = 1.5;
	double min_secs = 0.8, hang_secs = 0.8;
	int threads = 1, verbose = 0, endpoint_on = 1;

	for (int i = 4; i < argc; i++) {
		if (!strcmp(argv[i], "--mic") && i + 1 < argc) mic_dev = argv[++i];
		else if (!strcmp(argv[i], "--clip") && i + 1 < argc) clip_path = argv[++i];
		else if (!strcmp(argv[i], "--pad-sec") && i + 1 < argc) pad_sec = atof(argv[++i]);
		else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--threshold") && i + 1 < argc) threshold = atof(argv[++i]);
		else if (!strcmp(argv[i], "--gain") && i + 1 < argc) gain = atof(argv[++i]);
		else if (!strcmp(argv[i], "--cmd-secs") && i + 1 < argc) cmd_secs = atof(argv[++i]);
		else if (!strcmp(argv[i], "--cooldown") && i + 1 < argc) cooldown = atof(argv[++i]);
		else if (!strcmp(argv[i], "--min-secs") && i + 1 < argc) min_secs = atof(argv[++i]);
		else if (!strcmp(argv[i], "--hang-secs") && i + 1 < argc) hang_secs = atof(argv[++i]);
		else if (!strcmp(argv[i], "--no-endpoint")) endpoint_on = 0;
		else if (!strcmp(argv[i], "--verbose")) verbose = 1;
		else if (argv[i][0] != '-') wav_p = argv[i];
	}

	printf("oww_wake: TFLite %s, threads=%d\n", TfLiteVersion(), threads);

	Chain *c = malloc(sizeof(Chain));
	if (chain_init(c, mel_p, emb_p, wake_p, threads) != 0) { free(c); return 1; }

	int rc;
	if (mic_dev)
		rc = run_mic(c, mic_dev, threshold, gain, cmd_secs, cooldown, clip_path,
		             endpoint_on, min_secs, hang_secs);
	else if (wav_p)
		rc = run_file(c, wav_p, pad_sec, threshold, verbose);
	else { fprintf(stderr, "need a clip.wav (file mode) or --mic (mic mode)\n"); rc = 2; }

	chain_free(c);
	free(c);
	return rc;
}
