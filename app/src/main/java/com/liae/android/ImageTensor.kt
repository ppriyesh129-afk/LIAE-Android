package com.liae.android

import android.graphics.Bitmap
import android.graphics.Color

object ImageTensor {

    private const val SIZE = 128

    // Bitmap -> NHWC FloatArray (1x128x128x3)
    fun bitmapToTensor(bitmap: Bitmap): FloatArray {

        val resized = Bitmap.createScaledBitmap(bitmap, SIZE, SIZE, true)
        val tensor = FloatArray(SIZE * SIZE * 3)

        var i = 0

        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {

                val c = resized.getPixel(x, y)

                tensor[i++] = Color.red(c) / 255f
                tensor[i++] = Color.green(c) / 255f
                tensor[i++] = Color.blue(c) / 255f
            }
        }

        return tensor
    }

    // Planar RGB (3x128x128) -> Bitmap
    fun tensorToBitmap(tensor: FloatArray): Bitmap {

        val bmp = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)

        val plane = SIZE * SIZE

        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {

                val i = y * SIZE + x

                val r = (tensor[i].coerceIn(0f, 1f) * 255f).toInt()
                val g = (tensor[plane + i].coerceIn(0f, 1f) * 255f).toInt()
                val b = (tensor[plane * 2 + i].coerceIn(0f, 1f) * 255f).toInt()

                bmp.setPixel(x, y, Color.rgb(r, g, b))
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

                val v = (mask[i++].coerceIn(0f, 1f) * 255f).toInt()
                bmp.setPixel(x, y, Color.rgb(v, v, v))
            }
        }

        return bmp
    }
}
