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
      COLORREF clrBackGround = screen->pixels[x * SCRHEIGHT + y];
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

inline std::int64_t calcDiff(std::uint32_t src, std::uint32_t ref) {
    std::uint32_t r0 = (src >> 16) & 0xFF;
    std::uint32_t g0 = (src >> 8) & 0xFF;
    std::uint32_t b0 = src & 255;

    std::uint32_t r1 = ref >> 16;
    std::uint32_t g1 = (ref >> 8) & 0xFF;
    std::uint32_t b1 = ref & 0xFF;

    std::uint32_t dr = r0 - r1;
    std::uint32_t dg = g0 - g1;
    std::uint32_t db = b0 - b1;
    // calculate squared color difference;
    // take into account eye sensitivity to red, green and blue
    return 3 * dr * dr + 6 * dg * dg + db * db;
}
// -----------------------------------------------------------
// Fitness evaluation
// Compare current generation against reference image.
// -----------------------------------------------------------
std::int64_t Game::Evaluate() {
  constexpr uint count = SCRWIDTH * SCRHEIGHT;
  const auto& refpix = reference->pixels;
  const auto& pixels = screen->pixels;
  alignas(std::hardware_constructive_interference_size) std::int64_t diff = 0;
  for (std::uint32_t i = 0; i < count; i+=8) {
    diff += calcDiff(pixels[i],   refpix[i]);
    diff += calcDiff(pixels[i+1], refpix[i+1]);
    diff += calcDiff(pixels[i+2], refpix[i+2]);
    diff += calcDiff(pixels[i+3], refpix[i+3]);
    diff += calcDiff(pixels[i+4], refpix[i+4]);
    diff += calcDiff(pixels[i+5], refpix[i+5]);
    diff += calcDiff(pixels[i+6], refpix[i+6]);
    diff += calcDiff(pixels[i+7], refpix[i+7]);
  }
  return (diff >> 5);
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
