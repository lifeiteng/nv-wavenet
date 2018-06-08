/******************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <vector>
#include "matrix.h"
#include "nv_wavenet_reference.h"

Matrix* createMatrix(int r, int c) {
  float mean = 0.0;
  float scale = 0.5 / r;
  Matrix* m = new Matrix(r, c, false);
  m->randomize(mean, scale);
  return m;
}

template <typename T_weight, typename T_data, int R, int S, int A>
void runTest(int num_layers, int max_dilation, int batch_size, int num_iterations, int samples_per_iteration, int impl,
             bool inputsFromDevice = false, bool weightsFromDevice = false) {
  float mean = 0.0;
  float scale = 0.5 / R;

  // Just encode one-hot vector as an integer
  std::vector<int> yInPrev(batch_size);
  std::vector<int> yInCur(batch_size);

  for (int b = 0; b < batch_size; b++) {
    yInPrev[b] = rand() % A;
    yInCur[b] = rand() % A;
  }
  std::vector<int> yOut(batch_size);

  Matrix outputSelectors(batch_size, samples_per_iteration);
  outputSelectors.randomize(0.5, 1.0);

  Matrix embeddingsPrev(R, A, false);
  Matrix embeddingsCur(R, A, false);

  embeddingsPrev.randomize(mean, scale);
  embeddingsCur.randomize(mean, scale);

  std::vector<Matrix*> Wprev(num_layers);
  std::vector<Matrix*> Wcur(num_layers);
  std::vector<Matrix*> Bh(num_layers);
  std::vector<Matrix*> Wres(num_layers);
  std::vector<Matrix*> Bres(num_layers);
  std::vector<Matrix*> Wskip(num_layers);
  std::vector<Matrix*> Bskip(num_layers);
  std::vector<Matrix*> skipOut(num_layers + 1);

  // Retain results for dilated inputs
  std::vector<std::vector<Matrix*>> Xt(samples_per_iteration);
  for (int sample = 0; sample < samples_per_iteration; sample++) {
    Xt[sample].resize(num_layers + 1);
  }

  for (int l = 0; l < num_layers; l++) {
    // Weights
    Wprev[l] = createMatrix(2 * R, R);
    Wcur[l] = createMatrix(2 * R, R);
    Bh[l] = createMatrix(2 * R, 1);
    Wres[l] = createMatrix(R, R);
    Bres[l] = createMatrix(R, 1);
    Wskip[l] = createMatrix(S, R);
    Bskip[l] = createMatrix(S, 1);

    // Activations
    skipOut[l] = createMatrix(S, batch_size);
  }

  for (int sample = 0; sample < samples_per_iteration; sample++) {
    for (int layer = 0; layer < num_layers + 1; layer++) {
      Xt[sample][layer] = createMatrix(R, batch_size);
    }
  }

  Matrix WskipOut(A, S, false);
  WskipOut.randomize(mean, scale);
  Matrix BskipOut(A, 1, false);
  BskipOut.randomize(mean, scale);
  Matrix Wout(A, A, false);
  Wout.randomize(mean, scale);
  Matrix Bout(A, 1, false);
  Bout.randomize(mean, scale);

  Matrix skipOutFinal(A, batch_size, false);
  Matrix out(A, batch_size, false);
  Matrix p(A, batch_size, false);

  Matrix zero(S, batch_size, false);
  for (int row = 0; row < S; row++) {
    for (int col = 0; col < batch_size; col++) {
      zero.set(row, col, 0.f);
    }
  }

  nvWavenetReference ref(num_layers, batch_size, samples_per_iteration, R, S, A, max_dilation);

  ref.setEmbeddings(embeddingsPrev.data(), embeddingsCur.data());
  for (int l = 0; l < num_layers; l++) {
    ref.setLayerWeights(l, Wprev[l]->data(), Wcur[l]->data(), Bh[l]->data(), Wres[l]->data(), Bres[l]->data(),
                        Wskip[l]->data(), Bskip[l]->data());
  }
  ref.setOutWeights(WskipOut.data(), BskipOut.data(), Wout.data(), Bout.data());

  Matrix Lh(2 * R, samples_per_iteration * num_layers * batch_size);
  assert(Lh.data());
  Lh.randomize(mean, scale);
  ref.setInputs(Lh.data(), outputSelectors.data());

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < num_iterations; i++) {
    // printf("Iteration: %d\n", i);
    // Run reference implementation
    int* refYout = (int*)malloc(samples_per_iteration * batch_size * sizeof(int));
    ref.run(samples_per_iteration, batch_size, refYout);
  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << "Samples " << samples_per_iteration << " Time  "
    << diff.count() / num_iterations << " s\n\n";
}

int main(int argc, char* argv[]) {
  int num_layers = 24;
  int MAX_DILATION = 6;
  int batch_size = 4;
  int SAMPLES_PER_ITERATION = 24000;

  if (argc > 1) num_layers = atoi(argv[1]);
  if (argc > 2) MAX_DILATION = atoi(argv[2]);
  if (argc > 3) batch_size = atoi(argv[3]);
  if (argc > 4) SAMPLES_PER_ITERATION = atoi(argv[4]);

  srand(3);

  printf("Testing R=32, S=128\n");
  runTest<float, float, 32, 128, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 1);

  printf("Testing R=64, S=128\n");
  runTest<float, float, 64, 128, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 1, true, false);

  printf("Testing R=64, S=256\n");
  runTest<float, float, 64, 256, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 1);

  printf("Testing R=128, S=256\n");
  runTest<float, float, 128, 256, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 1);

  printf("Testing R=256, S=512\n");
  runTest<float, float, 256, 512, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 1);
}
