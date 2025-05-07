#include "game.h"
#include "precomp.h"
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <new>
#include <xmmintrin.h>

#define LINES 750
#define LINEFILE "lines750.dat"
#define ITERATIONS 16

int lx1[LINES], ly1[LINES], lx2[LINES], ly2[LINES]; // lines: start and end coordinates
uint lc[LINES];                                     // lines: colors
int x1_, y1_, x2_, y2_;                             // room for storing line backup
uint c_;                                            // line color backup
std::int64_t fitness;                                        // similarity to reference image
int lidx = 0;                                       // current line to be mutated
float peak = 0;                                     // peak line rendering performance
Surface *reference, *backup;                        // surfaces
Timer timer;

#define BYTE unsigned char
#define DWORD unsigned int
#define COLORREF DWORD
#define __int64 long long
#define RGB(r, g, b) (((DWORD)(BYTE)r) | ((DWORD)((BYTE)g) << 8) | ((DWORD)((BYTE)b) << 16))

#define GetRValue(RGBColor) (std::uint8_t)(RGBColor)
#define GetGValue(RGBColor) (std::uint8_t)((((uint)RGBColor) >> 8) & 0xFF)
#define GetBValue(RGBColor) (std::uint8_t)((((uint)RGBColor) >> 16) & 0xFF)

// -----------------------------------------------------------
// Mutate
// Randomly modify or replace one line.
// -----------------------------------------------------------
void MutateLine(int i) {
  // backup the line before modifying it
  x1_ = lx1[i], y1_ = ly1[i];
  x2_ = lx2[i], y2_ = ly2[i];
  c_ = lc[i];
  do {
    if (rand() & 1) {
      // color mutation (50% probability)
      lc[i] = RandomUInt() & 0xffffff;
    } else if (rand() & 1) {
      // small mutation (25% probability)
      lx1[i] += RandomUInt() % 6 - 3, ly1[i] += RandomUInt() % 6 - 3;
      lx2[i] += RandomUInt() % 6 - 3, ly2[i] += RandomUInt() % 6 - 3;
      // ensure the line stays on the screen
      lx1[i] = min(SCRWIDTH - 1, max(0, lx1[i]));
      lx2[i] = min(SCRWIDTH - 1, max(0, lx2[i]));
      ly1[i] = min(SCRHEIGHT - 1, max(0, ly1[i]));
      ly2[i] = min(SCRHEIGHT - 1, max(0, ly2[i]));
    } else {
      // new line (25% probability)
      lx1[i] = RandomUInt() % SCRWIDTH, lx2[i] = RandomUInt() % SCRWIDTH;
      ly1[i] = RandomUInt() % SCRHEIGHT, ly2[i] = RandomUInt() % SCRHEIGHT;
    }
  } while ((abs(lx1[i] - lx2[i]) < 3) || (abs(ly1[i] - ly2[i]) < 3));
}

void UndoMutation(int i) {
  // restore line i to the backuped state
  lx1[i] = x1_, ly1[i] = y1_;
  lx2[i] = x2_, ly2[i] = y2_;
  lc[i] = c_;
}

inline void plotLine(Surface *screen, std::uint32_t x, std::uint32_t y, std::int32_t rl, std::int32_t gl, std::int32_t bl, std::uint32_t grayl, std::uint32_t weight) {
      COLORREF clrBackGround = screen->pixels[x + y * SCRWIDTH];
      std::int32_t rb = GetRValue(clrBackGround);
      std::int32_t gb = GetGValue(clrBackGround);
      std::int32_t bb = GetBValue(clrBackGround);
      std::uint8_t grayb = (77 * rl + 150 * gl + 29 * bl) >> 8;
      // 0 or 255
      std::uint8_t mask = -(std::uint8_t)(grayl < grayb);
      std::uint32_t val = (weight ^ mask) >> 8;

      BYTE rr = (val * std::abs(rb - rl) + rl);
      BYTE gr = (val * std::abs(gb - gl) + gl);
      BYTE br = (val * std::abs(bb - bl) + bl);
      screen->Plot(x, y, RGB(rr, gr, br));
}
// -----------------------------------------------------------
// DrawWuLine
// Anti-aliased line rendering.
// Straight from:
// https://www.codeproject.com/Articles/13360/Antialiasing-Wu-Algorithm
// -----------------------------------------------------------
void DrawWuLine(Surface *screen, std::int32_t X0, std::int32_t Y0,
                std::int32_t X1, std::int32_t Y1, std::int32_t lineColor) {
  /* Make sure the line runs top to bottom */
  if (Y0 > Y1) {
    std::swap(Y0, Y1);
    std::swap(X0, X1);
  }

  /* Draw the initial pixel, which is always exactly intersected by
  the line and so needs no weighting */
  screen->Plot(X0, Y0, lineColor);

  std::int32_t XDir, DeltaX = X1 - X0;
  if (DeltaX >= 0) {
    XDir = 1;
  } else {
    XDir = -1;
    DeltaX = std::abs(DeltaX);
  }

  /* Special-case horizontal, vertical, and diagonal lines, which
  require no weighting because they go right through the center of
  every pixel */
  std::int32_t DeltaY = Y1 - Y0;

  unsigned short ErrorAdj;
  unsigned short ErrorAccTemp, weight;

  /* Line is not horizontal, diagonal, or vertical */
  unsigned short ErrorAcc = 0; /* initialize the line error accumulator to 0 */

  std::int32_t rl = GetRValue(lineColor);
  std::int32_t gl = GetGValue(lineColor);
  std::int32_t bl = GetBValue(lineColor);
  std::uint8_t grayl = (77 * rl + 150 * gl + 29 * bl) >> 8;

  /* Is this an X-major or Y-major line? */
  if (DeltaY > DeltaX) {
    /* Y-major line; calculate 16-bit fixed-point fractional part of a
    pixel that X advances each time Y advances 1 pixel, truncating the
        result so that we won't overrun the endpoint along the X axis */
    ErrorAdj = ((unsigned long)DeltaX << 16) / (unsigned long)DeltaY;

    /* Draw all pixels other than the first and last */
    while (--DeltaY) {
      ErrorAccTemp = ErrorAcc; /* remember currrent accumulated error */
      ErrorAcc += ErrorAdj;    /* calculate error for next pixel */
      X0 += (ErrorAcc <= ErrorAccTemp) * XDir;
      Y0++; /* Y-major, so always advance Y */

      /* The IntensityBits most significant bits of ErrorAcc give us the
      intensity weighting for this pixel, and the complement of the
      weighting for the paired pixel */
      weight = ErrorAcc >> 8;
      plotLine(screen, X0, Y0, rl, gl, bl, grayl, weight);
      plotLine(screen, X0, Y0 + XDir, rl, gl, bl, grayl, weight);
    }
    /* Draw the final pixel, which is always exactly intersected by the line
    and so needs no weighting */
    screen->Plot(X1, Y1, lineColor);
    return;
  }
  /* It's an X-major line; calculate 16-bit fixed-point fractional part of a
  pixel that Y advances each time X advances 1 pixel, truncating the
  result to avoid overrunning the endpoint along the X axis */
  ErrorAdj = ((unsigned long)DeltaY << 16) / (unsigned long)DeltaX;
  /* Draw all pixels other than the first and last */
  while (--DeltaX) {
    ErrorAccTemp = ErrorAcc; /* remember currrent accumulated error */
    ErrorAcc += ErrorAdj;    /* calculate error for next pixel */
    Y0 += ErrorAcc <= ErrorAccTemp;
    X0 += XDir; /* X-major, so always advance X */
                /* The IntensityBits most significant bits of ErrorAcc give us the
                intensity weighting for this pixel, and the complement of the
    weighting for the paired pixel */
    weight = ErrorAcc >> 8;

    plotLine(screen, X0, Y0, rl, gl, bl, grayl, weight);
    plotLine(screen, X0, Y0 + 1, rl, gl, bl, grayl, weight);
  }

  /* Draw the final pixel, which is always exactly intersected by the line
  and so needs no weighting */
  screen->Plot(X1, Y1, lineColor);
}
/// Computes the following:
/// diff = c0 - c1
/// diff *= diff
///
static inline __m256i compute_pow(__m256i c0, __m256i c1) {
    __m256i diff = _mm256_sub_epi32(c0, c1);
    __m256i pow = _mm256_mullo_epi32(diff, diff);
    return pow;
}

static inline __m256i manual(__m256i src, __m256i ref) {
    __m128i m = _mm_cvtsi32_si128(0xFF);
    __m256i mask = _mm256_broadcastd_epi32(m);

    __m256i r0 = _mm256_srli_epi32(src, 16);
    r0 = _mm256_and_si256(r0, mask);
    __m256i g0 = _mm256_srli_epi32(src, 8);
    g0 = _mm256_and_si256(g0, mask);
    __m256i b0 = _mm256_and_si256(src, mask);


    __m256i r1 = _mm256_srli_epi32(ref, 16);
    r1 = _mm256_and_si256(r1, mask);
    __m256i g1 = _mm256_srli_epi32(ref, 8);
    g1 = _mm256_and_si256(g1, mask);
    __m256i b1 = _mm256_and_si256(ref, mask);

  // dr = 3 * (r0-r1)^2
  __m256i dr2  = compute_pow(r0, r1);
  __m256i dr2a = _mm256_slli_epi32(dr2, 1);        // 2·dr²
  __m256i dr   = _mm256_add_epi32(dr2, dr2a);      // 3·dr²

  // dg = 6 * (g0-g1)^2
  __m256i dg2  = compute_pow(g0, g1);
  __m256i dg4  = _mm256_slli_epi32(dg2, 2);        // 4·dg²
  __m256i dg2a = _mm256_slli_epi32(dg2, 1);        // 2·dg²
  __m256i dg   = _mm256_add_epi32(dg4, dg2a);      // 6·dg²

    __m256i db = compute_pow(b0, b1);

    return _mm256_add_epi32(_mm256_add_epi32(dr, dg), db);
}

std::uint64_t Game::Evaluate() {
    const __m256i * __restrict__ refpix = (__m256i *) reference->pixels;
    const __m256i * __restrict__ pixels = (__m256i *) screen->pixels;
    __m256i diff = _mm256_set1_epi64x(0);

    constexpr std::uint32_t count = SCRWIDTH * SCRHEIGHT / 8;
    for (std::uint32_t i = 0; i < count; i++) {
        if ((i * 8) % 1024 == 0) {
            _mm_prefetch(&refpix[i+1024], _MM_HINT_T0);
            _mm_prefetch(&pixels[i+1024], _MM_HINT_T0);
        }
        __m256i ref = _mm256_load_si256(&refpix[i]);
        __m256i src = _mm256_load_si256(&pixels[i]);

        __m256i d32 = manual(src, ref);
        __m128i lo32 = _mm256_castsi256_si128(d32);
        __m128i hi32 = _mm256_extracti128_si256(d32, 1);

        // widen each to four 64-bit lanes
        __m256i lo64 = _mm256_cvtepu32_epi64(lo32);
        __m256i hi64 = _mm256_cvtepu32_epi64(hi32);

        // accumulate
        diff = _mm256_add_epi64(diff, lo64);
        diff = _mm256_add_epi64(diff, hi64);
    }
    alignas(32) uint64_t tmp[4];
    _mm256_store_si256((__m256i*)tmp, diff);
    uint64_t total = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    return (total >> 5);
}

// -----------------------------------------------------------
// Application initialization
// Load a previously saved generation, if available.
// -----------------------------------------------------------
void Game::Init() {
  for (int i = 0; i < LINES; i++)
    MutateLine(i);
  FILE *f = fopen(LINEFILE, "rb");
  if (f) {
    fread(lx1, 4, LINES, f);
    fread(ly1, 4, LINES, f);
    fread(lx2, 4, LINES, f);
    fread(ly2, 4, LINES, f);
    fread(lc, 4, LINES, f);
    fclose(f);
  }
  reference = new Surface("assets/bird.png");
  backup = new Surface(SCRWIDTH, SCRHEIGHT);
  memset(screen->pixels, 255, SCRWIDTH * SCRHEIGHT * 4);
  for (int j = 0; j < LINES; j++) {
    DrawWuLine(screen, lx1[j], ly1[j], lx2[j], ly2[j], lc[j]);
  }
  fitness = Evaluate();
}

// -----------------------------------------------------------
// Main application tick function
// -----------------------------------------------------------
void Game::Tick(float /* deltaTime */) {
  timer.reset();
  int lineCount = 0;
  int iterCount = 0;
  // draw up to lidx
  memset(screen->pixels, 255, SCRWIDTH * SCRHEIGHT * 4);
  for (int j = 0; j < lidx; j+=2, lineCount+=2) {
    DrawWuLine(screen, lx1[j], ly1[j], lx2[j], ly2[j], lc[j]);
    DrawWuLine(screen, lx1[j+1], ly1[j+1], lx2[j+1], ly2[j+1], lc[j+1]);
  }
  int base = lidx;
  screen->CopyTo(backup, 0, 0);
  // iterate and draw from lidx to end
  for (int k = 0; k < ITERATIONS; k++) {
    backup->CopyTo(screen, 0, 0);
    MutateLine(lidx);
    for (int j = base; j < LINES; j+=2, lineCount+=2) {
      DrawWuLine(screen, lx1[j], ly1[j], lx2[j], ly2[j], lc[j]);
      DrawWuLine(screen, lx1[j+1], ly1[j+1], lx2[j+1], ly2[j+1], lc[j+1]);
    }
    std::int64_t diff = Evaluate();
    if (diff < fitness)
      fitness = diff;
    else
      UndoMutation(lidx);
    lidx = (lidx + 1) % LINES;
    iterCount++;
  }
  // stats
  char t[128];
  float elapsed = timer.elapsed();
  float lps = (float)lineCount / elapsed;
  peak = max(lps, peak);
  sprintf(t, "fitness: %i", fitness);
  screen->Print(t, 2, SCRHEIGHT - 72, 0xFF);
  sprintf(t, "lps:     %5.2fK", lps);
  screen->Print(t, 2, SCRHEIGHT - 54, 0xFF);
  sprintf(t, "ips:     %5.2f", (iterCount * 1000) / elapsed);
  screen->Print(t, 2, SCRHEIGHT - 36, 0xFF);
  sprintf(t, "peak:    %5.2f", peak);
  screen->Print(t, 2, SCRHEIGHT - 18, 0xFF);
}

// -----------------------------------------------------------
// Application termination
// Save the current generation, so we can continue later.
// -----------------------------------------------------------
void Game::Shutdown() {
  FILE *f = fopen(LINEFILE, "wb");
  fwrite(lx1, 4, LINES, f);
  fwrite(ly1, 4, LINES, f);
  fwrite(lx2, 4, LINES, f);
  fwrite(ly2, 4, LINES, f);
  fwrite(lc, 4, LINES, f);
  fclose(f);
}
