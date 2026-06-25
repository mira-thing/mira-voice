// persistent on-device ASR sidecar

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr,
            "usage: %s <encoder> <decoder> <joiner> <tokens> [num_threads] [decoding_method]\n",
            argv[0]);
    return 2;
  }
  const char *encoder = argv[1];
  const char *decoder = argv[2];
  const char *joiner = argv[3];
  const char *tokens = argv[4];
  int num_threads = (argc > 5) ? atoi(argv[5]) : 4;
  const char *method = (argc > 6) ? argv[6] : "greedy_search";

  // line buffer stdout so each transcript flushes over the pipe
  setvbuf(stdout, NULL, _IOLBF, 0);

  SherpaOnnxOfflineRecognizerConfig config;
  memset(&config, 0, sizeof(config));
  config.feat_config.sample_rate = 16000;
  config.feat_config.feature_dim = 80;
  config.model_config.transducer.encoder = encoder;
  config.model_config.transducer.decoder = decoder;
  config.model_config.transducer.joiner = joiner;
  config.model_config.tokens = tokens;
  config.model_config.num_threads = num_threads;
  config.model_config.provider = "cpu";
  config.model_config.debug = 0;
  config.decoding_method = method;
  config.max_active_paths = 4;

  fprintf(stderr, "sherpa_asr_server: loading %s (threads=%d, method=%s)...\n", encoder,
          num_threads, method);
  const SherpaOnnxOfflineRecognizer *rec = SherpaOnnxCreateOfflineRecognizer(&config);
  if (rec == NULL) {
    fprintf(stderr, "sherpa_asr_server: failed to create recognizer\n");
    return 3;
  }
  fprintf(stderr, "sherpa_asr_server: model loaded, ready\n");
  printf("READY\n");
  fflush(stdout);

  char line[8192];
  while (fgets(line, sizeof(line), stdin) != NULL) {
    size_t n = strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
    if (n == 0) {
      printf("\n");
      fflush(stdout);
      continue;
    }

    const SherpaOnnxWave *wave = SherpaOnnxReadWave(line);
    if (wave == NULL) {
      fprintf(stderr, "sherpa_asr_server: read wave failed: %s\n", line);
      printf("\n");
      fflush(stdout);
      continue;
    }

    const SherpaOnnxOfflineStream *stream = SherpaOnnxCreateOfflineStream(rec);
    SherpaOnnxAcceptWaveformOffline(stream, wave->sample_rate, wave->samples,
                                    wave->num_samples);
    SherpaOnnxDecodeOfflineStream(rec, stream);
    const SherpaOnnxOfflineRecognizerResult *res = SherpaOnnxGetOfflineStreamResult(stream);

    const char *text = (res != NULL && res->text != NULL) ? res->text : "";
    for (const char *p = text; *p != '\0'; ++p) putchar(*p == '\n' ? ' ' : *p);
    putchar('\n');
    fflush(stdout);

    SherpaOnnxDestroyOfflineRecognizerResult(res);
    SherpaOnnxDestroyOfflineStream(stream);
    SherpaOnnxFreeWave(wave);
  }

  SherpaOnnxDestroyOfflineRecognizer(rec);
  return 0;
}
