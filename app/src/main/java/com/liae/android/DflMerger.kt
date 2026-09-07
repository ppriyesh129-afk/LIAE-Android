package com.liae.android

import android.graphics.Bitmap
import android.graphics.Color

object DflMerger {

    private const val MASK_DILATION = 4
    private const val BLUR_RADIUS = 6

    fun merge(
        background: Bitmap,
        warpedFace: Bitmap,
        warpedMask: Bitmap
    ): Bitmap {

        require(background.width == warpedFace.width)
        require(background.height == warpedFace.height)
        require(background.width == warpedMask.width)
        require(background.height == warpedMask.height)

        val dilated = dilateMask(warpedMask)
        val feathered = featherMask(dilated)
        dilated.recycle()

        val corrected = colorTransfer(
            warpedFace,
            background,
            feathered
        )

        val output = alphaBlend(
            background,
            corrected,
            feathered
        )

        corrected.recycle()
        feathered.recycle()

        return output
    }

    private fun dilateMask(mask: Bitmap): Bitmap {

        val w = mask.width
        val h = mask.height

        val src = IntArray(w * h)
        val out = IntArray(w * h)

        mask.getPixels(src, 0, w, 0, 0, w, h)

        for (y in 0 until h) {
            for (x in 0 until w) {

                var best = 0

                for (dy in -MASK_DILATION..MASK_DILATION) {
                    for (dx in -MASK_DILATION..MASK_DILATION) {

                        val nx = x + dx
                        val ny = y + dy

                        if (nx !in 0 until w) continue
                        if (ny !in 0 until h) continue

                        val alpha = src[ny * w + nx] ushr 24

                        if (alpha > best)
                            best = alpha
                    }
                }

                out[y * w + x] =
                    (best shl 24) or 0x00FFFFFF
            }
        }

        return Bitmap.createBitmap(
            out,
            w,
            h,
            Bitmap.Config.ARGB_8888
        )
    }

    private fun featherMask(mask: Bitmap): Bitmap {

        val w = mask.width
        val h = mask.height

        val src = IntArray(w * h)
        val tmp = IntArray(w * h)
        val out = IntArray(w * h)

        mask.getPixels(src, 0, w, 0, 0, w, h)

        // Horizontal blur

        for (y in 0 until h) {
            for (x in 0 until w) {

                var sum = 0
                var count = 0

                for (dx in -BLUR_RADIUS..BLUR_RADIUS) {

                    val nx = x + dx

                    if (nx !in 0 until w)
                        continue

                    sum += src[y * w + nx] ushr 24
                    count++
                }

                val alpha = sum / count

                tmp[y * w + x] =
                    (alpha shl 24) or 0x00FFFFFF
            }
        }

        // Vertical blur

        for (y in 0 until h) {
            for (x in 0 until w) {

                var sum = 0
                var count = 0

                for (dy in -BLUR_RADIUS..BLUR_RADIUS) {

                    val ny = y + dy

                    if (ny !in 0 until h)
                        continue

                    sum += tmp[ny * w + x] ushr 24
                    count++
                }

                val alpha = sum / count

                out[y * w + x] =
                    (alpha shl 24) or 0x00FFFFFF
            }
        }

        return Bitmap.createBitmap(
            out,
            w,
            h,
            Bitmap.Config.ARGB_8888
        )
    }

    private fun colorTransfer(
        face: Bitmap,
        background: Bitmap,
        mask: Bitmap
    ): Bitmap {

        val w = face.width
        val h = face.height

        val fp = IntArray(w * h)
        val bp = IntArray(w * h)
        val mp = IntArray(w * h)

        face.getPixels(fp, 0, w, 0, 0, w, h)
        background.getPixels(bp, 0, w, 0, 0, w, h)
        mask.getPixels(mp, 0, w, 0, 0, w, h)

        var fr = 0f
        var fg = 0f
        var fb = 0f

        var br = 0f
        var bg = 0f
        var bb = 0f

        var count = 0

        for (i in fp.indices) {

            if ((mp[i] ushr 24) < 20)
                continue

            fr += Color.red(fp[i])
            fg += Color.green(fp[i])
            fb += Color.blue(fp[i])

            br += Color.red(bp[i])
            bg += Color.green(bp[i])
            bb += Color.blue(bp[i])

            count++
        }

        if (count == 0)
            return face.copy(Bitmap.Config.ARGB_8888, true)

        val rScale =
            (br / fr.coerceAtLeast(1f))
                .coerceIn(0.5f, 2f)

        val gScale =
            (bg / fg.coerceAtLeast(1f))
                .coerceIn(0.5f, 2f)

        val bScale =
            (bb / fb.coerceAtLeast(1f))
                .coerceIn(0.5f, 2f)

        val out = IntArray(w * h)

        for (i in fp.indices) {

            val r =
                (Color.red(fp[i]) * rScale)
                    .toInt()
                    .coerceIn(0, 255)

            val g =
                (Color.green(fp[i]) * gScale)
                    .toInt()
                    .coerceIn(0, 255)

            val b =
                (Color.blue(fp[i]) * bScale)
                    .toInt()
                    .coerceIn(0, 255)

            out[i] =
                Color.argb(255, r, g, b)
        }

        return Bitmap.createBitmap(
            out,
            w,
            h,
            Bitmap.Config.ARGB_8888
        )
    }

    private fun alphaBlend(
        background: Bitmap,
        foreground: Bitmap,
        mask: Bitmap
    ): Bitmap {

        val w = background.width
        val h = background.height

        val bg = IntArray(w * h)
        val fg = IntArray(w * h)
        val mk = IntArray(w * h)
        val out = IntArray(w * h)

        background.getPixels(bg, 0, w, 0, 0, w, h)
        foreground.getPixels(fg, 0, w, 0, 0, w, h)
        mask.getPixels(mk, 0, w, 0, 0, w, h)

        for (i in out.indices) {

            val alpha = (mk[i] ushr 24) / 255f

            val r =
                (Color.red(bg[i]) * (1f - alpha) +
                        Color.red(fg[i]) * alpha)
                    .toInt()
                    .coerceIn(0, 255)

            val g =
                (Color.green(bg[i]) * (1f - alpha) +
                        Color.green(fg[i]) * alpha)
                    .toInt()
                    .coerceIn(0, 255)

            val b =
                (Color.blue(bg[i]) * (1f - alpha) +
                        Color.blue(fg[i]) * alpha)
                    .toInt()
                    .coerceIn(0, 255)

            out[i] =
                Color.argb(255, r, g, b)
        }

        return Bitmap.createBitmap(
            out,
            w,
            h,
            Bitmap.Config.ARGB_8888
        )
    }
}
