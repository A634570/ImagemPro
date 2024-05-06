package com.android.example.imagempro

import android.graphics.Bitmap
import android.util.Base64
import androidx.camera.core.ImageProxy
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.imgcodecs.Imgcodecs

object ImgUtil {

    fun toBase64(image: Mat): String {
        val matOfByte = MatOfByte()
        Imgcodecs.imencode(".png", image, matOfByte)
        val byteArray = matOfByte.toArray()
        return Base64.encodeToString(byteArray, Base64.DEFAULT)
    }

    fun convertImageProxyToMat(image: ImageProxy): Mat {
        val bitmap = image.toBitmap()
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        return mat
    }

    fun convertMatToBitmap(image: Mat): Bitmap {
        val bitmap = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(image, bitmap)
        return bitmap
    }
}
