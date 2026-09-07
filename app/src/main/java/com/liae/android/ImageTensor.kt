package com.liae.android

import android.graphics.Bitmap
import android.graphics.Color

object ImageTensor {

    private const val SIZE = 128

    /*
     * Bitmap -> NHWC FloatArray
     * Shape:
     * [1,128,128,3]
     */
    fun bitmapToTensor(bitmap: Bitmap): FloatArray {

        val resized =
            Bitmap.createScaledBitmap(
                bitmap,
                SIZE,
                SIZE,
                true
            )

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
            FloatArray(SIZE * SIZE * 3)

        var i = 0

        for (pixel in pixels) {

            tensor[i++] =
                Color.red(pixel) / 255f

            tensor[i++] =
                Color.green(pixel) / 255f

            tensor[i++] =
                Color.blue(pixel) / 255f
        }

        if (resized !== bitmap) {
            resized.recycle()
        }

        return tensor
    }

    /*
     * NHWC FloatArray -> Bitmap
     * Shape:
     * [1,128,128,3]
     */
    fun tensorToBitmap(tensor: FloatArray): Bitmap {

        require(
            tensor.size == SIZE * SIZE * 3
        ) {
            "Expected ${SIZE * SIZE * 3} floats, got ${tensor.size}"
        }

        val pixels =
            IntArray(SIZE * SIZE)

        var i = 0

        for (p in pixels.indices) {

            val r =
                (tensor[i++]
                    .coerceIn(0f, 1f) * 255f)
                    .toInt()

            val g =
                (tensor[i++]
                    .coerceIn(0f, 1f) * 255f)
                    .toInt()

            val b =
                (tensor[i++]
                    .coerceIn(0f, 1f) * 255f)
                    .toInt()

            pixels[p] =
                Color.rgb(r, g, b)
        }

        return Bitmap.createBitmap(
            pixels,
            SIZE,
            SIZE,
            Bitmap.Config.ARGB_8888
        )
    }

    /*
     * Mask -> Grayscale Bitmap
     * Shape:
     * [1,128,128,1]
     */
    fun maskToBitmap(mask: FloatArray): Bitmap {

        require(
            mask.size == SIZE * SIZE
        ) {
            "Expected ${SIZE * SIZE} mask values, got ${mask.size}"
        }

        val pixels =
            IntArray(SIZE * SIZE)

        for (i in pixels.indices) {

            val v =
                (mask[i]
                    .coerceIn(0f, 1f) * 255f)
                    .toInt()

            pixels[i] =
                Color.rgb(v, v, v)
        }

        return Bitmap.createBitmap(
            pixels,
            SIZE,
            SIZE,
            Bitmap.Config.ARGB_8888
        )
    }
}
