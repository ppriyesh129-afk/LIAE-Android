package com.liae.android

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

object LandmarksProcessor {

    private const val SIZE = 128

    // Reference landmark positions for a 128x128 aligned face.
    private val target = MatOfPoint2f(
        Point(43.77, 59.08),   // left eye
        Point(84.04, 58.86),   // right eye
        Point(64.03, 81.98),   // nose
        Point(47.48,105.56),   // left mouth
        Point(80.83,105.38)    // right mouth
    )

    data class WarpResult(
        val bitmap: Bitmap,
        val matrix: Mat
    )

    fun warpFace(bitmap: Bitmap, landmarks: Array<Point>): WarpResult {

        val src = Mat()
        Utils.bitmapToMat(bitmap, src)

        val srcPts = MatOfPoint2f(*landmarks)

        val matrix = Imgproc.getAffineTransform(
            MatOfPoint2f(
                landmarks[0],
                landmarks[1],
                landmarks[2]
            ),
            MatOfPoint2f(
                target.toArray()[0],
                target.toArray()[1],
                target.toArray()[2]
            )
        )

        val aligned = Mat()

        Imgproc.warpAffine(
            src,
            aligned,
            matrix,
            Size(SIZE.toDouble(), SIZE.toDouble()),
            Imgproc.INTER_LINEAR
        )

        val out = Bitmap.createBitmap(
            SIZE,
            SIZE,
            Bitmap.Config.ARGB_8888
        )

        Utils.matToBitmap(aligned, out)

        src.release()
        aligned.release()

        return WarpResult(out, matrix)
    }

    fun inverseWarp(
        face: Bitmap,
        matrix: Mat,
        width: Int,
        height: Int
    ): Bitmap {

        val src = Mat()
        Utils.bitmapToMat(face, src)

        val inv = Mat()
        Imgproc.invertAffineTransform(matrix, inv)

        val out = Mat()

        Imgproc.warpAffine(
            src,
            out,
            inv,
            Size(width.toDouble(), height.toDouble()),
            Imgproc.INTER_LINEAR,
            Core.BORDER_TRANSPARENT
        )

        val bmp = Bitmap.createBitmap(
            width,
            height,
            Bitmap.Config.ARGB_8888
        )

        Utils.matToBitmap(out, bmp)

        src.release()
        out.release()
        inv.release()

        return bmp
    }
}
