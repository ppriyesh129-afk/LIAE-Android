package com.liae.android

import android.graphics.Bitmap
import android.graphics.Color

object ImageTensor {

    fun bitmapToTensor(bitmap: Bitmap): FloatArray {

        val resized = Bitmap.createScaledBitmap(
            bitmap,
            128,
            128,
            true
        )

        val pixels = IntArray(128 * 128)

        resized.getPixels(
            pixels,
            0,
            128,
            0,
            0,
            128,
            128
        )

        val tensor = FloatArray(128 * 128 * 3)

        var index = 0

        for (y in 0 until 128) {
            for (x in 0 until 128) {

                val pixel = pixels[y * 128 + x]

                tensor[index++] = Color.red(pixel) / 255.0f
                tensor[index++] = Color.green(pixel) / 255.0f
                tensor[index++] = Color.blue(pixel) / 255.0f
            }
        }

        if (resized !== bitmap) {
            resized.recycle()
        }

        return tensor
    }

    fun tensorToBitmap(
        tensor: FloatArray
    ): Bitmap {

        require(tensor.size == 128 * 128 * 3) {
            "Expected 128x128x3 tensor"
        }

        val pixels = IntArray(128 * 128)

        var index = 0

        for (y in 0 until 128) {
            for (x in 0 until 128) {

                val r = (tensor[index++]
                    .coerceIn(0.0f, 1.0f) * 255.0f).toInt()

                val g = (tensor[index++]
                    .coerceIn(0.0f, 1.0f) * 255.0f).toInt()

                val b = (tensor[index++]
                    .coerceIn(0.0f, 1.0f) * 255.0f).toInt()

                pixels[y * 128 + x] =
                    Color.rgb(r, g, b)
            }
        }

        return Bitmap.createBitmap(
            pixels,
            128,
            128,
            Bitmap.Config.ARGB_8888
        )
    }
}
