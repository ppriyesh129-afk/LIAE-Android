package com.liae.android

import android.graphics.Bitmap
import android.graphics.Color

object ImageTensor {

    private const val SIZE = 128

    // Bitmap -> NCHW FloatArray (1x3x128x128)
    fun bitmapToTensor(bitmap: Bitmap): FloatArray {

        val resized = Bitmap.createScaledBitmap(bitmap, SIZE, SIZE, true)

        val tensor = FloatArray(3 * SIZE * SIZE)

        var i = 0
        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {

                val c = resized.getPixel(x, y)

                tensor[i] = Color.red(c) / 255f
                tensor[SIZE * SIZE + i] = Color.green(c) / 255f
                tensor[2 * SIZE * SIZE + i] = Color.blue(c) / 255f

                i++
            }
        }

        return tensor
    }

    // NCHW FloatArray -> Bitmap
    fun tensorToBitmap(tensor: FloatArray): Bitmap {

        val bmp = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)

        var i = 0
        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {

                val r = (tensor[i].coerceIn(0f, 1f) * 255f).toInt()
                val g = (tensor[SIZE * SIZE + i].coerceIn(0f, 1f) * 255f).toInt()
                val b = (tensor[2 * SIZE * SIZE + i].coerceIn(0f, 1f) * 255f).toInt()

                bmp.setPixel(x, y, Color.rgb(r, g, b))
                i++
            }
        }

        return bmp
    }

    // Mask -> grayscale Bitmap
    fun maskToBitmap(mask: FloatArray): Bitmap {

        val bmp = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)

        var i = 0
        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {

                val v = (mask[i].coerceIn(0f, 1f) * 255f).toInt()
                bmp.setPixel(x, y, Color.rgb(v, v, v))
                i++
            }
        }

        return bmp
    }
}
