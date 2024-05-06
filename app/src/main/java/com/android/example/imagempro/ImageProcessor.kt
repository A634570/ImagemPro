package com.android.example.imagempro

import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageProxy
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

object ImageProcessor {

    fun processImage(image: ImageProxy): Bitmap {
        val matImage = ImgUtil.convertImageProxyToMat(image)

        val grayMat = grayImage(matImage)
        val grayImage = ImgUtil.convertMatToBitmap(grayMat)

        val blurredMat = blurImage(grayMat)
        val blurredImage = ImgUtil.convertMatToBitmap(blurredMat)

        val openedMat = applyOpening(blurredMat)
        val openedImage = ImgUtil.convertMatToBitmap(openedMat)

        val thresholdMat = thresholdImage(openedMat)
        val thresholdImage = ImgUtil.convertMatToBitmap(thresholdMat)

        val cannyMat = cannyImage(thresholdMat)
        val cannyImage = ImgUtil.convertMatToBitmap(cannyMat)

        val countours = findCountours(cannyMat)
        val maxCountour = findLargestContour(countours)

        val drawedMat = drawContour(matImage, maxCountour)
        val drawedImage = ImgUtil.convertMatToBitmap(drawedMat)

        val birdseyeMat = birdseyeView(matImage, maxCountour)
        val birdseyeImage = ImgUtil.convertMatToBitmap(birdseyeMat)

        Log.d("A", "SUCURRU")

        return ImgUtil.convertMatToBitmap(birdseyeMat)
    }


    private fun grayImage(image: Mat): Mat {
        val grayImage = Mat()
        Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_RGBA2GRAY)
        return grayImage
    }

    private fun blurImage(image: Mat): Mat {
        val blurredImage = Mat()
        Imgproc.GaussianBlur(image, blurredImage, Size(25.0, 25.0), 1.0)
        return blurredImage
    }

    private fun thresholdImage(image: Mat): Mat {
        val thresholdImage = Mat()
        Imgproc.adaptiveThreshold(
            image,
            thresholdImage,
            255.0,
            Imgproc.ADAPTIVE_THRESH_MEAN_C,
            Imgproc.THRESH_BINARY_INV,
            11,
            4.0
        )
        return thresholdImage
    }

    private fun cannyImage(image: Mat): Mat {
        val cannyImage = Mat()
        Imgproc.Canny(image, cannyImage, 50.0, 150.0, 5, false)
        return cannyImage
    }

    private fun applyOpening(image: Mat): Mat {
        val openedImage = Mat()
        Imgproc.morphologyEx(
            image,
            openedImage,
            Imgproc.MORPH_OPEN,
            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        )
        return openedImage
    }

    private fun findCountours(image: Mat): List<MatOfPoint> {
        val contours: List<MatOfPoint> = mutableListOf()
        val hierarchy = Mat()
        Imgproc.findContours(
            image,
            contours,
            hierarchy,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )
        return contours.sortedByDescending { contour -> Imgproc.contourArea(contour) }
    }

    private fun findLargestContour(contours: List<MatOfPoint>): MatOfPoint {
        var maxArea = 0.0
        var maxContour = MatOfPoint()
        val matOfPoint2f = MatOfPoint2f()

        for (contour in contours) {
            val contourArea = Imgproc.contourArea(contour)
            if (contourArea > 10000) {
                contour.convertTo(matOfPoint2f, CvType.CV_32F)

                val peri = Imgproc.arcLength(matOfPoint2f, true)
                val approx = MatOfPoint2f()
                Imgproc.approxPolyDP(matOfPoint2f, approx, 0.02 * peri, true)

                if (contourArea > maxArea && approx.rows() == 4) {
                    maxContour = contour
                    maxArea = contourArea
                }
            }
        }
        return maxContour
    }

    private fun drawContour(image: Mat, contour: MatOfPoint): Mat {
        val copyFrame = Mat()
        image.copyTo(copyFrame)

        if (contour.empty()) {
            return copyFrame
        }

        Imgproc.drawContours(copyFrame, mutableListOf(contour), -1, Scalar(0.0, 0.0, 255.0), 4)
        return copyFrame
    }

    private fun birdseyeView(image: Mat, contour: MatOfPoint): Mat {
        val birdseyeImage = Mat()
        image.copyTo(birdseyeImage)

        if (contour.empty()) {
            return birdseyeImage
        }

        val auxMat = MatOfPoint2f()
        contour.convertTo(auxMat, CvType.CV_32F)
        val epsilon = 0.02 * Imgproc.arcLength(auxMat, true)
        val approxCurve = MatOfPoint2f()
        Imgproc.approxPolyDP(auxMat, approxCurve, epsilon, true)

        val outputWidth = 480
        val outputHeight = 640

        val destinationPoints = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(outputWidth.toDouble(), 0.0),
            Point(outputWidth.toDouble(), outputHeight.toDouble()),
            Point(0.0, outputHeight.toDouble())
        )

        val transformationMatrix = Imgproc.getPerspectiveTransform(
            approxCurve,
            destinationPoints
        )

        Imgproc.warpPerspective(
            image,
            birdseyeImage,
            transformationMatrix,
            Size(outputWidth.toDouble(), outputHeight.toDouble())
        )

        return birdseyeImage
    }

}