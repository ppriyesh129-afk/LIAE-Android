package com.liae.android

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

object DflAligner {

    private const val SIZE = 128

    /*
     * DeepFaceLab FULL-face padding.
     *
     * This controls the amount of face context included
     * around the aligned face.
     */
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

        require(source.width > 0) {
            "Invalid source bitmap width"
        }

        require(source.height > 0) {
            "Invalid source bitmap height"
        }

        require(face.keypoints.size >= 4) {
            "BlazeFace landmarks are missing"
        }

        /*
         * BlazeFace landmarks:
         *
         * 0 = right eye
         * 1 = left eye
         * 2 = nose
         * 3 = mouth
         */
        val src =
            arrayOf(

                Point2(
                    face.keypoints[0][0],
                    face.keypoints[0][1]
                ),

                Point2(
                    face.keypoints[1][0],
                    face.keypoints[1][1]
                ),

                Point2(
                    face.keypoints[2][0],
                    face.keypoints[2][1]
                ),

                Point2(
                    face.keypoints[3][0],
                    face.keypoints[3][1]
                )
            )

        /*
         * Canonical 128x128 face coordinates.
         *
         * These provide a stable face coordinate system
         * for the LIAE input.
         */
        val dst =
            canonicalPoints()

        /*
         * Estimate rotation + scale + translation.
         *
         * This is much better than using the detector
         * bounding box as a square crop.
         */
        val forward =
            estimateSimilarityTransform(
                src,
                dst
            )

        val inverse =
            invertAffine(
                forward
            )

        val matrix =
            Matrix()

        matrix.setValues(
            floatArrayOf(

                forward[0],
                forward[1],
                forward[2],

                forward[3],
                forward[4],
                forward[5],

                0f,
                0f,
                1f
            )
        )

        val aligned =
            Bitmap.createBitmap(
                SIZE,
                SIZE,
                Bitmap.Config.ARGB_8888
            )

        val canvas =
            Canvas(aligned)

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

    private fun canonicalPoints():
            Array<Point2> {

        /*
         * Canonical face positions in the
         * 128x128 neural-network coordinate space.
         *
         * This is a BlazeFace-compatible approximation
         * of the DFL FULL-face coordinate system.
         *
         * It is NOT the original DFL 68-point transform,
         * because BlazeFace supplies only 6 keypoints.
         */

        val centerX = 64f
        val centerY = 64f

        val points =
            arrayOf(

                // Right eye
                Point2(
                    89f,
                    43f
                ),

                // Left eye
                Point2(
                    39f,
                    43f
                ),

                // Nose
                Point2(
                    64f,
                    65f
                ),

                // Mouth
                Point2(
                    64f,
                    88f
                )
            )

        /*
         * Apply the DFL FULL-face padding
         * to the canonical coordinates.
         */
        val paddingScale =
            1f /
                    (
                        1f +
                                DFL_PADDING * 2f
                        )

        return points.map { point ->

            Point2(

                centerX +
                        (
                            point.x -
                                    centerX
                            ) *
                        paddingScale,

                centerY +
                        (
                            point.y -
                                    centerY
                            ) *
                        paddingScale
            )

        }.toTypedArray()
    }

    private fun estimateSimilarityTransform(
        src: Array<Point2>,
        dst: Array<Point2>
    ): FloatArray {

        require(src.size == dst.size) {
            "Source/destination landmark count mismatch"
        }

        require(src.size >= 3) {
            "At least 3 landmarks are required"
        }

        var srcCx = 0.0
        var srcCy = 0.0

        var dstCx = 0.0
        var dstCy = 0.0

        for (i in src.indices) {

            srcCx +=
                src[i].x

            srcCy +=
                src[i].y

            dstCx +=
                dst[i].x

            dstCy +=
                dst[i].y
        }

        val count =
            src.size.toDouble()

        srcCx /= count
        srcCy /= count

        dstCx /= count
        dstCy /= count

        var a = 0.0
        var b = 0.0
        var denominator = 0.0

        for (i in src.indices) {

            val sx =
                src[i].x.toDouble() -
                        srcCx

            val sy =
                src[i].y.toDouble() -
                        srcCy

            val dx =
                dst[i].x.toDouble() -
                        dstCx

            val dy =
                dst[i].y.toDouble() -
                        dstCy

            a +=
                sx * dx +
                        sy * dy

            b +=
                sx * dy -
                        sy * dx

            denominator +=
                sx * sx +
                        sy * sy
        }

        require(
            denominator > 1e-8
        ) {
            "Degenerate BlazeFace landmarks"
        }

        val scale =
            sqrt(
                a * a +
                        b * b
            ) /
                    denominator

        val angle =
            atan2(
                b,
                a
            )

        val c =
            cos(angle) *
                    scale

        val s =
            sin(angle) *
                    scale

        val tx =
            dstCx -
                    (
                        c * srcCx -
                                s * srcCy
                        )

        val ty =
            dstCy -
                    (
                        s * srcCx +
                                c * srcCy
                        )

        return floatArrayOf(

            c.toFloat(),
            (-s).toFloat(),
            tx.toFloat(),

            s.toFloat(),
            c.toFloat(),
            ty.toFloat()
        )
    }

    private fun invertAffine(
        matrix: FloatArray
    ): FloatArray {

        val a =
            matrix[0]

        val b =
            matrix[1]

        val tx =
            matrix[2]

        val c =
            matrix[3]

        val d =
            matrix[4]

        val ty =
            matrix[5]

        val determinant =
            a * d -
                    b * c

        require(
            kotlin.math.abs(
                determinant
            ) > 1e-8f
        ) {
            "Non-invertible face transform"
        }

        val invA =
            d /
                    determinant

        val invB =
            -b /
                    determinant

        val invC =
            -c /
                    determinant

        val invD =
            a /
                    determinant

        val invTx =
            -(
                invA * tx +
                        invB * ty
                )

        val invTy =
            -(
                invC * tx +
                        invD * ty
                )

        return floatArrayOf(

            invA,
            invB,
            invTx,

            invC,
            invD,
            invTy
        )
    }

    private data class Point2(
        val x: Float,
        val y: Float
    )
}
