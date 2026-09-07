package com.liae.android

import android.graphics.Bitmap
import android.graphics.Color

object ImageTensor {

    private const val SIZE = 128
    private const val CHANNELS = 3

    fun bitmapToTensor(bitmap: Bitmap): FloatArray {

        val resized = Bitmap.createScaledBitmap(
            bitmap,
            SIZE,
            SIZE,
            true
        )

        val pixels = IntArray(SIZE * SIZE)

        resized.getPixels(
            pixels,
            0,
            SIZE,
            0,
            0,
            SIZE,
            SIZE
        )

        val tensor = FloatArray(
            SIZE * SIZE * CHANNELS
        )

        var index = 0

        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {

                val pixel = pixels[
                    y * SIZE + x
                ]

                tensor[index++] =
                    Color.red(pixel) / 255.0f

                tensor[index++] =
                    Color.green(pixel) / 255.0f

                tensor[index++] =
                    Color.blue(pixel) / 255.0f
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

        require(
            tensor.size ==
                SIZE * SIZE * CHANNELS
        ) {
            "Expected 128x128x3 tensor, got ${tensor.size}"
        }

        val pixels = IntArray(
            SIZE * SIZE
        )

        var index = 0

        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {

                val r =
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

                val b =
                    (
                        tensor[index++]
                            .coerceIn(0.0f, 1.0f) *
                            255.0f
                    ).toInt()

                pixels[
                    y * SIZE + x
                ] = Color.rgb(
                    r,
                    g,
                    b
                )
            }
        }

        return Bitmap.createBitmap(
            pixels,
            SIZE,
            SIZE,
            Bitmap.Config.ARGB_8888
        )
    }
}
