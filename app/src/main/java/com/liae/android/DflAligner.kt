package com.liae.android

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import kotlin.math.max

object DflAligner {

    private const val SIZE = 128

    // DeepFaceLab FULL face padding.
    private const val DFL_PADDING = 0.2109375f

    data class AlignedFace(
        val bitmap: Bitmap,
        val forward: FloatArray,
        val inverse: FloatArray
    )

    fun align(
        source: Bitmap,
        face: BlazeFaceResult
    ): AlignedFace {

        require(source.width > 0 && source.height > 0)

        val faceCx = (face.left + face.right) * 0.5f
        val faceCy = (face.top + face.bottom) * 0.5f

        val faceSize = max(
            face.right - face.left,
            face.bottom - face.top
        )

        val cropSize =
            faceSize * (1f + DFL_PADDING * 2f)

        val scale =
            SIZE.toFloat() / cropSize

        val tx =
            SIZE / 2f - faceCx * scale

        val ty =
            SIZE / 2f - faceCy * scale

        val forward = floatArrayOf(
            scale, 0f, tx,
            0f, scale, ty
        )

        val inverse = floatArrayOf(
            1f / scale,
            0f,
            -tx / scale,

            0f,
            1f / scale,
            -ty / scale
        )

        val matrix = Matrix()

        matrix.setValues(
            floatArrayOf(
                forward[0], forward[1], forward[2],
                forward[3], forward[4], forward[5],
                0f, 0f, 1f
            )
        )

        val aligned =
            Bitmap.createBitmap(
                SIZE,
                SIZE,
                Bitmap.Config.ARGB_8888
            )

        val canvas = Canvas(aligned)

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG
            )

        canvas.drawBitmap(
            source,
            matrix,
            paint
        )

        return AlignedFace(
            bitmap = aligned,
            forward = forward,
            inverse = inverse
        )
    }
}
