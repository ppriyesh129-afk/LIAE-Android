package com.liae.android

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.objdetect.FaceDetectorYN

class FaceDetector(modelPath: String) {

    private val detector = FaceDetectorYN.create(
        modelPath,
        "",
        Size(320.0,320.0)
    )

    data class Face(
        val rect: Rect,
        val landmarks: Array<Point>
    )

    fun detect(bitmap: Bitmap): Face? {

        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)

        detector.setInputSize(mat.size())

        val faces = Mat()

        detector.detect(mat, faces)

        if (faces.rows() == 0)
            return null

        val row = faces.row(0)

        val x = row.get(0,0)[0].toInt()
        val y = row.get(0,1)[0].toInt()
        val w = row.get(0,2)[0].toInt()
        val h = row.get(0,3)[0].toInt()

        val pts = Array(5){
            Point(
                row.get(0,4+it*2)[0],
                row.get(0,5+it*2)[0]
            )
        }

        return Face(
            Rect(x,y,w,h),
            pts
        )
    }
}
