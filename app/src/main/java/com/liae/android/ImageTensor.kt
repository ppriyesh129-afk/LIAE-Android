package com.liae.android

import android.graphics.Bitmap
import android.graphics.Color

object ImageTensor {

    private const val SIZE = 128
    private const val CHANNELS = 3

    /**
     * Converts any Bitmap into a 128x128 float tensor (NHWC layout).
     * Values are normalized to 0.0..1.0.
     */
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

        val tensor = FloatArray(SIZE * SIZE * CHANNELS)

        var p = 0
        var i = 0

        while (i < pixels.size) {

            val c = pixels[i]

            tensor[p++] = Color.red(c) / 255f
            tensor[p++] = Color.green(c) / 255f
            tensor[p++] = Color.blue(c) / 255f

            i++
        }

        if (resized != bitmap) {
            resized.recycle()
        }

        return tensor
    }

    /**
     * Converts the ONNX RGB output back into a Bitmap.
     * Handles values in both 0..1 and -1..1 ranges.
     */
    fun tensorToBitmap(rgb: FloatArray): Bitmap {

        require(rgb.size >= SIZE * SIZE * CHANNELS) {
            "RGB tensor is too small: ${rgb.size}"
        }

        val bitmap = Bitmap.createBitmap(
            SIZE,
            SIZE,
            Bitmap.Config.ARGB_8888
        )

        val pixels = IntArray(SIZE * SIZE)

        var p = 0

        for (i in pixels.indices) {

            var r = rgb[p++]
            var g = rgb[p++]
            var b = rgb[p++]

            // Support both [-1,1] and [0,1] model outputs.
            if (r < 0f || g < 0f || b < 0f) {
                r = (r + 1f) * 0.5f
                g = (g + 1f) * 0.5f
                b = (b + 1f) * 0.5f
            }

            val rr = (r.coerceIn(0f, 1f) * 255f).toInt()
            val gg = (g.coerceIn(0f, 1f) * 255f).toInt()
            val bb = (b.coerceIn(0f, 1f) * 255f).toInt()

            pixels[i] = Color.argb(
                255,
                rr,
                gg,
                bb
            )
        }

        bitmap.setPixels(
            pixels,
            0,
            SIZE,
            0,
            0,
            SIZE,
            SIZE
        )

        return bitmap
    }

    /**
     * Converts a mask tensor into a grayscale Bitmap for debugging.
     */
    fun maskToBitmap(mask: FloatArray): Bitmap {

        require(mask.size >= SIZE * SIZE) {
            "Mask tensor is too small: ${mask.size}"
        }

        val bitmap = Bitmap.createBitmap(
            SIZE,
            SIZE,
            Bitmap.Config.ARGB_8888
        )

        val pixels = IntArray(SIZE * SIZE)

        for (i in pixels.indices) {

            var v = mask[i]

            if (v < 0f) {
                v = (v + 1f) * 0.5f
            }

            val gray =
                (v.coerceIn(0f, 1f) * 255f).toInt()

            pixels[i] = Color.argb(
                255,
                gray,
                gray,
                gray
            )
        }

        bitmap.setPixels(
            pixels,
            0,
            SIZE,
            0,
            0,
            SIZE,
            SIZE
        )

        return bitmap
    }
}
