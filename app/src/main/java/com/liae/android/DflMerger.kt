package com.liae.android

import android.graphics.Bitmap
import android.graphics.Color

object DflMerger {

    fun merge(
        background: Bitmap,
        warpedFace: Bitmap,
        warpedMask: Bitmap
    ): Bitmap {

        require(background.width == warpedFace.width)
        require(background.height == warpedFace.height)
        require(background.width == warpedMask.width)
        require(background.height == warpedMask.height)

        val w = background.width
        val h = background.height

        val backgroundPixels = IntArray(w * h)
        val facePixels = IntArray(w * h)
        val maskPixels = IntArray(w * h)
        val outputPixels = IntArray(w * h)

        background.getPixels(backgroundPixels, 0, w, 0, 0, w, h)
        warpedFace.getPixels(facePixels, 0, w, 0, 0, w, h)
        warpedMask.getPixels(maskPixels, 0, w, 0, 0, w, h)

        for (i in outputPixels.indices) {

            val maskAlpha =
                (Color.red(maskPixels[i]) / 255f)
                    .coerceIn(0f, 1f)

            when {
                maskAlpha <= 0f -> {
                    outputPixels[i] = backgroundPixels[i]
                }

                maskAlpha >= 1f -> {
                    outputPixels[i] = facePixels[i]
                }

                else -> {

                    val bgPixel = backgroundPixels[i]
                    val facePixel = facePixels[i]

                    val inv = 1f - maskAlpha

                    val r = (
                        Color.red(bgPixel) * inv +
                        Color.red(facePixel) * maskAlpha
                    ).toInt().coerceIn(0, 255)

                    val g = (
                        Color.green(bgPixel) * inv +
                        Color.green(facePixel) * maskAlpha
                    ).toInt().coerceIn(0, 255)

                    val b = (
                        Color.blue(bgPixel) * inv +
                        Color.blue(facePixel) * maskAlpha
                    ).toInt().coerceIn(0, 255)

                    outputPixels[i] = Color.argb(255, r, g, b)
                }
            }
        }

        return Bitmap.createBitmap(
            outputPixels,
            w,
            h,
            Bitmap.Config.ARGB_8888
        )
    }
}
