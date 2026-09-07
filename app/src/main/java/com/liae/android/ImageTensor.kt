package com.liae.android

import android.graphics.Bitmap
import android.graphics.Color

object ImageTensor {

    private const val SIZE = 128
    private const val CHANNELS = 3

    private const val IMAGE_FLOATS =
        SIZE * SIZE * CHANNELS

    private const val MASK_FLOATS =
        SIZE * SIZE

    /*
     * Android Bitmap RGB
     *
     * ->
     *
     * LIAE NHWC BGR
     *
     * Shape: [1,128,128,3]
     * Range: 0.0 .. 1.0
     */
    fun bitmapToTensor(
        bitmap: Bitmap
    ): FloatArray {

        val resized =
            if (
                bitmap.width == SIZE &&
                bitmap.height == SIZE
            ) {
                bitmap
            } else {
                Bitmap.createScaledBitmap(
                    bitmap,
                    SIZE,
                    SIZE,
                    true
                )
            }

        try {

            val pixels =
                IntArray(SIZE * SIZE)

            resized.getPixels(
                pixels,
                0,
                SIZE,
                0,
                0,
                SIZE,
                SIZE
            )

            val tensor =
                FloatArray(IMAGE_FLOATS)

            var index = 0

            /*
             * NHWC
             *
             * B
             * G
             * R
             */
            for (pixel in pixels) {

                tensor[index++] =
                    Color.blue(pixel) / 255.0f

                tensor[index++] =
                    Color.green(pixel) / 255.0f

                tensor[index++] =
                    Color.red(pixel) / 255.0f
            }

            return tensor

        } finally {

            if (
                resized !== bitmap &&
                !resized.isRecycled
            ) {
                resized.recycle()
            }
        }
    }

    /*
     * LIAE output
     *
     * Shape: [1,128,128,3]
     * Layout: BGR
     * Range: 0.0 .. 1.0
     *
     * Converts back to Android RGB Bitmap.
     */
    fun tensorToBitmap(
        tensor: FloatArray
    ): Bitmap {

        require(
            tensor.size == IMAGE_FLOATS
        ) {
            "Expected $IMAGE_FLOATS values, " +
                "got ${tensor.size}"
        }

        val pixels =
            IntArray(SIZE * SIZE)

        var index = 0

        for (i in pixels.indices) {

            val b =
                (
                    tensor[index++]
                        .coerceIn(0.0f, 1.0f) *
                        255.0f
                ).toInt()

            val g =
                (
                    tensor[index++]
                        .coerceIn(0.0f, 1.0f) *
                        255.0f
                ).toInt()

            val r =
                (
                    tensor[index++]
                        .coerceIn(0.0f, 1.0f) *
                        255.0f
                ).toInt()

            pixels[i] =
                Color.rgb(
                    r,
                    g,
                    b
                )
        }

        return Bitmap.createBitmap(
            pixels,
            SIZE,
            SIZE,
            Bitmap.Config.ARGB_8888
        )
    }

    /*
     * LIAE mask
     *
     * Shape: [1,128,128,1]
     *
     * Store mask as normal grayscale RGB.
     *
     * R = mask
     * G = mask
     * B = mask
     * A = 255
     */
    fun maskToBitmap(
        mask: FloatArray
    ): Bitmap {

        require(
            mask.size == MASK_FLOATS
        ) {
            "Expected $MASK_FLOATS mask values, " +
                "got ${mask.size}"
        }

        val pixels =
            IntArray(MASK_FLOATS)

        for (i in pixels.indices) {

            val value =
                (
                    mask[i]
                        .coerceIn(0.0f, 1.0f) *
                        255.0f
                ).toInt()

            pixels[i] =
                Color.argb(
                    255,
                    value,
                    value,
                    value
                )
        }

        return Bitmap.createBitmap(
            pixels,
            SIZE,
            SIZE,
            Bitmap.Config.ARGB_8888
        )
    }
}
