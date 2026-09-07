package com.liae.android

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import kotlin.math.max
import kotlin.math.min

class FaceSwapPipeline(
    private val detector: YuNetDetector,
    private val liae: LiaeUdEngine
) {

    data class Result(
        val bitmap: Bitmap,
        val facesDetected: Int
    )

    fun swap(
        sourceImage: Bitmap,
        targetImage: Bitmap
    ): Result {

        val sourceFaces =
            detector.detect(sourceImage)

        val targetFaces =
            detector.detect(targetImage)

        if (sourceFaces.isEmpty()) {
            throw IllegalStateException(
                "No face detected in source image"
            )
        }

        if (targetFaces.isEmpty()) {
            throw IllegalStateException(
                "No face detected in target image"
            )
        }

        /*
         * Use the highest-confidence source face.
         */
        val sourceFace =
            sourceFaces.maxByOrNull {
                it.score
            } ?: throw IllegalStateException(
                "Source face selection failed"
            )

        /*
         * Start with the original target.
         */
        var result =
            targetImage.copy(
                Bitmap.Config.ARGB_8888,
                true
            )

        /*
         * Process every detected target face.
         */
        for (targetFace in targetFaces) {

            result =
                swapOneFace(
                    sourceImage = sourceImage,
                    sourceFace = sourceFace,
                    targetImage = result,
                    targetFace = targetFace
                )
        }

        return Result(
            bitmap = result,
            facesDetected = targetFaces.size
        )
    }

    private fun swapOneFace(
        sourceImage: Bitmap,
        sourceFace: YuNetDetector.Face,
        targetImage: Bitmap,
        targetFace: YuNetDetector.Face
    ): Bitmap {

        /*
         * --------------------------------------------------
         * SOURCE ALIGNMENT
         * --------------------------------------------------
         */

        val alignedSource =
            DflAligner.align(
                sourceImage,
                sourceFace
            )

        /*
         * --------------------------------------------------
         * TARGET ALIGNMENT
         * --------------------------------------------------
         */

        val alignedTarget =
            DflAligner.align(
                targetImage,
                targetFace
            )

        try {

            /*
             * Convert both aligned faces to
             * NHWC float tensors.
             */
            val sourceTensor =
                ImageTensor.bitmapToTensor(
                    alignedSource.bitmap
                )

            val targetTensor =
                ImageTensor.bitmapToTensor(
                    alignedTarget.bitmap
                )

            /*
             * --------------------------------------------------
             * LIAE-UD
             * --------------------------------------------------
             *
             * Inputs:
             *
             * src [1,128,128,3]
             * dst [1,128,128,3]
             *
             * Outputs:
             *
             * RGB  [1,128,128,3]
             * MASK [1,128,128,1]
             */

            val prediction =
                liae.run(
                    src = sourceTensor,
                    dst = targetTensor
                )

            /*
             * Convert generated RGB face
             * into a bitmap.
             */
            val swappedFace =
                ImageTensor.tensorToBitmap(
                    prediction.rgb
                )

            /*
             * Convert the single-channel LIAE mask
             * into a usable alpha mask.
             */
            val mask =
                createMaskBitmap(
                    prediction.mask
                )

            try {

                /*
                 * --------------------------------------------------
                 * INVERSE AFFINE
                 * --------------------------------------------------
                 *
                 * The generated 128x128 face and mask are currently
                 * in aligned DFL coordinates.
                 *
                 * Warp them back into the target image.
                 */

                val warpedFace =
                    warpToTarget(
                        alignedFace = swappedFace,
                        inverse = alignedTarget.inverse,
                        targetWidth = targetImage.width,
                        targetHeight = targetImage.height
                    )

                val warpedMask =
                    warpToTarget(
                        alignedFace = mask,
                        inverse = alignedTarget.inverse,
                        targetWidth = targetImage.width,
                        targetHeight = targetImage.height
                    )

                /*
                 * --------------------------------------------------
                 * BLEND
                 * --------------------------------------------------
                 */

                val output =
                    blend(
                        background = targetImage,
                        foreground = warpedFace,
                        mask = warpedMask
                    )

                warpedFace.recycle()
                warpedMask.recycle()

                return output

            } finally {

                swappedFace.recycle()
                mask.recycle()
            }

        } finally {

            alignedSource.bitmap.recycle()
            alignedTarget.bitmap.recycle()
        }
    }

    /**
     * Convert LIAE's [128,128,1] mask into
     * an ARGB bitmap.
     */
    private fun createMaskBitmap(
        mask: FloatArray
    ): Bitmap {

        val expected =
            128 * 128

        require(mask.size == expected) {
            "Expected 128x128 mask, got ${mask.size}"
        }

        val pixels =
            IntArray(expected)

        for (i in 0 until expected) {

            /*
             * LIAE mask is expected in 0..1.
             */
            val alpha =
                (
                    mask[i]
                        .coerceIn(0.0f, 1.0f) *
                        255.0f
                    ).toInt()

            pixels[i] =
                (
                    alpha shl 24
                ) or
                    0x00FFFFFF
        }

        return Bitmap.createBitmap(
            pixels,
            128,
            128,
            Bitmap.Config.ARGB_8888
        )
    }

    /**
     * Warp a 128x128 aligned image back to
     * original target-image coordinates.
     *
     * DflAligner.inverse maps:
     *
     * aligned -> original
     */
    private fun warpToTarget(
        alignedFace: Bitmap,
        inverse: FloatArray,
        targetWidth: Int,
        targetHeight: Int
    ): Bitmap {

        val matrix =
            Matrix()

        matrix.setValues(
            floatArrayOf(
                inverse[0],
                inverse[1],
                inverse[2],

                inverse[3],
                inverse[4],
                inverse[5],

                0f,
                0f,
                1f
            )
        )

        val output =
            Bitmap.createBitmap(
                targetWidth,
                targetHeight,
                Bitmap.Config.ARGB_8888
            )

        val canvas =
            Canvas(output)

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                    Paint.FILTER_BITMAP_FLAG
            )

        canvas.drawBitmap(
            alignedFace,
            matrix,
            paint
        )

        return output
    }

    /**
     * Alpha blend:
     *
     * output =
     *     foreground * mask +
     *     background * (1-mask)
     *
     * This runs only where the warped mask
     * contains useful values.
     */
    private fun blend(
        background: Bitmap,
        foreground: Bitmap,
        mask: Bitmap
    ): Bitmap {

        require(
            background.width ==
                foreground.width &&
                background.height ==
                foreground.height
        )

        require(
            background.width ==
                mask.width &&
                background.height ==
                mask.height
        )

        val width =
            background.width

        val height =
            background.height

        val bgPixels =
            IntArray(width * height)

        val fgPixels =
            IntArray(width * height)

        val maskPixels =
            IntArray(width * height)

        background.getPixels(
            bgPixels,
            0,
            width,
            0,
            0,
            width,
            height
        )

        foreground.getPixels(
            fgPixels,
            0,
            width,
            0,
            0,
            width,
            height
        )

        mask.getPixels(
            maskPixels,
            0,
            width,
            0,
            0,
            width,
            height
        )

        val output =
            IntArray(width * height)

        for (i in output.indices) {

            val bg =
                bgPixels[i]

            val fg =
                fgPixels[i]

            /*
             * Alpha channel from mask.
             */
            val alpha =
                (
                    maskPixels[i] ushr 24
                ) / 255.0f

            /*
             * Slight feathering.
             *
             * This prevents a hard edge around the
             * generated face.
             */
            val a =
                smoothStep(
                    0.05f,
                    0.95f,
                    alpha
                )

            val br =
                (bg shr 16) and 0xFF

            val bgc =
                (bg shr 8) and 0xFF

            val bb =
                bg and 0xFF

            val fr =
                (fg shr 16) and 0xFF

            val fgc =
                (fg shr 8) and 0xFF

            val fb =
                fg and 0xFF

            val r =
                (
                    br * (1.0f - a) +
                        fr * a
                    ).toInt()
                        .coerceIn(0, 255)

            val g =
                (
                    bgc * (1.0f - a) +
                        fgc * a
                    ).toInt()
                        .coerceIn(0, 255)

            val b =
                (
                    bb * (1.0f - a) +
                        fb * a
                    ).toInt()
                        .coerceIn(0, 255)

            output[i] =
                (
                    0xFF shl 24
                ) or
                    (r shl 16) or
                    (g shl 8) or
                    b
        }

        return Bitmap.createBitmap(
            output,
            width,
            height,
            Bitmap.Config.ARGB_8888
        )
    }

    private fun smoothStep(
        edge0: Float,
        edge1: Float,
        value: Float
    ): Float {

        val x =
            (
                (value - edge0) /
                    (edge1 - edge0)
                ).coerceIn(
                    0.0f,
                    1.0f
                )

        return x * x *
            (3.0f - 2.0f * x)
    }

    fun close() {
        detector.close()
        liae.close()
    }
}
