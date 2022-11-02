/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkBase64Utilities.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// Program below is modified from vtkBase64Utilities.cxx.

#include <cassert>
#include <fstream>
#include <sstream>

//----------------------------------------------------------------------------
static const unsigned char vtkBase64UtilitiesEncodeTable[65] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                                               "abcdefghijklmnopqrstuvwxyz"
                                                               "0123456789+/";

//----------------------------------------------------------------------------
inline static unsigned char vtkBase64UtilitiesEncodeChar(unsigned char c)
{
  assert(c < 65);
  return vtkBase64UtilitiesEncodeTable[c];
}

//----------------------------------------------------------------------------
void EncodeTriplet(unsigned char i0, unsigned char i1, unsigned char i2,
  unsigned char* o0, unsigned char* o1, unsigned char* o2, unsigned char* o3)
{
  *o0 = vtkBase64UtilitiesEncodeChar((i0 >> 2) & 0x3F);
  *o1 = vtkBase64UtilitiesEncodeChar(((i0 << 4) & 0x30) | ((i1 >> 4) & 0x0F));
  *o2 = vtkBase64UtilitiesEncodeChar(((i1 << 2) & 0x3C) | ((i2 >> 6) & 0x03));
  *o3 = vtkBase64UtilitiesEncodeChar(i2 & 0x3F);
}

//----------------------------------------------------------------------------
void EncodePair(unsigned char i0, unsigned char i1, unsigned char* o0,
  unsigned char* o1, unsigned char* o2, unsigned char* o3)
{
  *o0 = vtkBase64UtilitiesEncodeChar((i0 >> 2) & 0x3F);
  *o1 = vtkBase64UtilitiesEncodeChar(((i0 << 4) & 0x30) | ((i1 >> 4) & 0x0F));
  *o2 = vtkBase64UtilitiesEncodeChar(((i1 << 2) & 0x3C));
  *o3 = '=';
}

//----------------------------------------------------------------------------
void EncodeSingle(
  unsigned char i0, unsigned char* o0, unsigned char* o1, unsigned char* o2, unsigned char* o3)
{
  *o0 = vtkBase64UtilitiesEncodeChar((i0 >> 2) & 0x3F);
  *o1 = vtkBase64UtilitiesEncodeChar(((i0 << 4) & 0x30));
  *o2 = '=';
  *o3 = '=';
}

//----------------------------------------------------------------------------
unsigned long Encode(
  const unsigned char* input, unsigned long length, unsigned char* output, int mark_end)
{

  const unsigned char* ptr = input;
  const unsigned char* end = input + length;
  unsigned char* optr = output;

  // Encode complete triplet

  while ((end - ptr) >= 3)
  {
    EncodeTriplet(
      ptr[0], ptr[1], ptr[2], &optr[0], &optr[1], &optr[2], &optr[3]);
    ptr += 3;
    optr += 4;
  }

  // Encodes a 2-byte ending into 3 bytes and 1 pad byte and writes.

  if (end - ptr == 2)
  {
    EncodePair(ptr[0], ptr[1], &optr[0], &optr[1], &optr[2], &optr[3]);
    optr += 4;
  }

  // Encodes a 1-byte ending into 2 bytes and 2 pad bytes

  else if (end - ptr == 1)
  {
    EncodeSingle(ptr[0], &optr[0], &optr[1], &optr[2], &optr[3]);
    optr += 4;
  }

  // Do we need to mark the end

  else if (mark_end)
  {
    optr[0] = optr[1] = optr[2] = optr[3] = '=';
    optr += 4;
  }

  return optr - output;
}

//----------------------------------------------------------------------------
static const unsigned char vtkBase64UtilitiesDecodeTable[256] = {
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0x3E, 0xFF, 0xFF, 0xFF, 0x3F, //
  0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, //
  0x3C, 0x3D, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, //
  0xFF, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, //
  0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, //
  0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, //
  0x17, 0x18, 0x19, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, //
  0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, //
  0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30, //
  0x31, 0x32, 0x33, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  //-------------------------------------
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, //
  0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF  //
};

//----------------------------------------------------------------------------
inline static unsigned char vtkBase64UtilitiesDecodeChar(unsigned char c)
{
  return vtkBase64UtilitiesDecodeTable[c];
}

//----------------------------------------------------------------------------
int DecodeTriplet(unsigned char i0, unsigned char i1, unsigned char i2,
  unsigned char i3, unsigned char* o0, unsigned char* o1, unsigned char* o2)
{
  unsigned char d0, d1, d2, d3;

  d0 = vtkBase64UtilitiesDecodeChar(i0);
  d1 = vtkBase64UtilitiesDecodeChar(i1);
  d2 = vtkBase64UtilitiesDecodeChar(i2);
  d3 = vtkBase64UtilitiesDecodeChar(i3);

  // Make sure all characters were valid

  if (d0 == 0xFF || d1 == 0xFF || d2 == 0xFF || d3 == 0xFF)
  {
    return 0;
  }

  // Decode the 3 bytes

  *o0 = ((d0 << 2) & 0xFC) | ((d1 >> 4) & 0x03);
  *o1 = ((d1 << 4) & 0xF0) | ((d2 >> 2) & 0x0F);
  *o2 = ((d2 << 6) & 0xC0) | ((d3 >> 0) & 0x3F);

  // Return the number of bytes actually decoded

  if (i2 == '=')
  {
    return 1;
  }
  if (i3 == '=')
  {
    return 2;
  }
  return 3;
}

//----------------------------------------------------------------------------
size_t DecodeSafely(
  const unsigned char* input, size_t inputLen, unsigned char* output, size_t outputLen)
{
  assert(input);
  assert(output);

  // Nonsense small input or no space for any output
  if ((inputLen < 4) || (outputLen == 0))
  {
    return 0;
  }

  // Consume 4 ASCII chars of input at a time, until less than 4 left
  size_t inIdx = 0, outIdx = 0;
  while (inIdx <= inputLen - 4)
  {
    // Decode 4 ASCII characters into 0, 1, 2, or 3 bytes
    unsigned char o0, o1, o2;
    int bytesDecoded = DecodeTriplet(
      input[inIdx + 0], input[inIdx + 1], input[inIdx + 2], input[inIdx + 3], &o0, &o1, &o2);
    assert((bytesDecoded >= 0) && (bytesDecoded <= 3));

    if ((bytesDecoded >= 1) && (outIdx < outputLen))
    {
      output[outIdx++] = o0;
    }
    if ((bytesDecoded >= 2) && (outIdx < outputLen))
    {
      output[outIdx++] = o1;
    }
    if ((bytesDecoded >= 3) && (outIdx < outputLen))
    {
      output[outIdx++] = o2;
    }

    // If fewer than 3 bytes resulted from decoding (in this pass),
    // then the input stream has nothing else decodable, so end.
    if (bytesDecoded < 3)
    {
      return outIdx;
    }

    // Consumed a whole 4 of input and got 3 bytes of output, continue
    inIdx += 4;
    assert(bytesDecoded == 3);
  }

  return outIdx;
}

int main()
{

    return 0;
}
