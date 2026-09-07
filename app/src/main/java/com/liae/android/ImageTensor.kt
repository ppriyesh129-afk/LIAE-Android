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
     * Bitmap -> NHWC RGB tensor
     *
     * Shape:
     * [1, 128, 128, 3]
     *
     * Layout:
     * RGB RGB RGB ...
     *
     * Range:
     * 0.0 .. 1.0
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
                IntArray(
                    SIZE * SIZE
                )

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
                FloatArray(
                    IMAGE_FLOATS
                )

            var index = 0

            for (pixel in pixels) {

                tensor[index++] =
                    Color.red(pixel) / 255.0f

                tensor[index++] =
                    Color.green(pixel) / 255.0f

                tensor[index++] =
                    Color.blue(pixel) / 255.0f
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
     * DFL LIAE output -> Android Bitmap
     *
     * Model output is BGR.
     *
     * Tensor:
     * [1, 128, 128, 3]
     *
     * Tensor layout:
     * BGR BGR BGR ...
     *
     * Android Bitmap:
     * RGB
     */
    fun tensorToBitmap(
        tensor: FloatArray
    ): Bitmap {

        require(
            tensor.size == IMAGE_FLOATS
        ) {
            "Expected $IMAGE_FLOATS values, got ${tensor.size}"
        }

        val pixels =
            IntArray(
                SIZE * SIZE
            )

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
     * LIAE mask -> Bitmap
     *
     * Model:
     * [1, 128, 128, 1]
     *
     * Mask value is stored in Alpha.
     */
    fun maskToBitmap(
        mask: FloatArray
    ): Bitmap {

        require(
            mask.size == MASK_FLOATS
        ) {
            "Expected $MASK_FLOATS mask values, got ${mask.size}"
        }

        val pixels =
            IntArray(
                MASK_FLOATS
            )

        for (i in pixels.indices) {

            val value =
                (
                    mask[i]
                        .coerceIn(0.0f, 1.0f) *
                    255.0f
                ).toInt()

            pixels[i] =
                Color.argb(
                    value,
                    255,
                    255,
                    255
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
