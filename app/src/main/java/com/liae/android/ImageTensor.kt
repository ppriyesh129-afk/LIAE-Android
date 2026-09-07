package com.liae.android

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.roundToInt

object ImageTensor {

    private const val SIZE = 128
    private const val CHANNELS = 3

    /*
     * Android Bitmap -> NHWC RGB float tensor.
     *
     * Output:
     * [1, 128, 128, 3]
     *
     * Values:
     * 0.0 .. 1.0
     */
    fun bitmapToTensor(bitmap: Bitmap): FloatArray {

        val resized =
            Bitmap.createScaledBitmap(
                bitmap,
                SIZE,
                SIZE,
                true
            )

        val pixels =
            IntArray(SIZE * SIZE)

        resized.getPixels(
            pixels,
            0,
            SIZE,
            0,
            0,
            SIZE,
            SIZE
        )

        val tensor =
            FloatArray(
                SIZE * SIZE * CHANNELS
            )

        var index = 0

        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {

                val pixel =
                    pixels[y * SIZE + x]

                tensor[index++] =
                    Color.red(pixel) / 255.0f

                tensor[index++] =
                    Color.green(pixel) / 255.0f

                tensor[index++] =
                    Color.blue(pixel) / 255.0f
            }
        }

        if (resized !== bitmap) {
            resized.recycle()
        }

        return tensor
    }

    /*
     * Converts LIAE RGB output to Bitmap.
     *
     * Handles:
     *
     * 0..1
     * -1..1
     *
     * and protects against NaN / Infinity.
     */
    fun modelRgbToBitmap(
        tensor: FloatArray,
        bgr: Boolean = false
    ): Bitmap {

        require(
            tensor.size ==
                SIZE * SIZE * CHANNELS
        ) {
            "Expected 128x128x3 RGB tensor, got ${tensor.size}"
        }

        var min =
            Float.POSITIVE_INFINITY

        var max =
            Float.NEGATIVE_INFINITY

        for (value in tensor) {

            if (value.isFinite()) {

                if (value < min) {
                    min = value
                }

                if (value > max) {
                    max = value
                }
            }
        }

        /*
         * LIAE output is normally expected to be
         * either 0..1 or -1..1.
         *
         * If negative values are present, assume
         * the model is using -1..1.
         */
        val minusOneToOne =
            min < -0.01f

        val pixels =
            IntArray(SIZE * SIZE)

        var index = 0

        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {

                var c0 =
                    tensor[index++]

                var c1 =
                    tensor[index++]

                var c2 =
                    tensor[index++]

                if (!c0.isFinite()) c0 = 0.0f
                if (!c1.isFinite()) c1 = 0.0f
                if (!c2.isFinite()) c2 = 0.0f

                if (minusOneToOne) {
                    c0 = (c0 + 1.0f) * 0.5f
                    c1 = (c1 + 1.0f) * 0.5f
                    c2 = (c2 + 1.0f) * 0.5f
                }

                c0 =
                    c0.coerceIn(
                        0.0f,
                        1.0f
                    )

                c1 =
                    c1.coerceIn(
                        0.0f,
                        1.0f
                    )

                c2 =
                    c2.coerceIn(
                        0.0f,
                        1.0f
                    )

                val r: Float
                val g: Float
                val b: Float

                if (bgr) {
                    b = c0
                    g = c1
                    r = c2
                } else {
                    r = c0
                    g = c1
                    b = c2
                }

                pixels[y * SIZE + x] =
                    Color.rgb(
                        (r * 255.0f).roundToInt(),
                        (g * 255.0f).roundToInt(),
                        (b * 255.0f).roundToInt()
                    )
            }
        }

        return Bitmap.createBitmap(
            pixels,
            SIZE,
            SIZE,
            Bitmap.Config.ARGB_8888
        )
    }

    /*
     * Applies the LIAE mask over the RGB result.
     *
     * mask is expected to contain 128x128 values.
     *
     * This keeps the generated face inside the
     * model's predicted face region.
     */
    fun applyMask(
        rgb: FloatArray,
        mask: FloatArray,
        background: FloatArray
    ): FloatArray {

        require(
            rgb.size ==
                SIZE * SIZE * CHANNELS
        )

        require(
            background.size ==
                SIZE * SIZE * CHANNELS
        )

        require(
            mask.size ==
                SIZE * SIZE
        )

        val result =
            FloatArray(rgb.size)

        for (i in 0 until SIZE * SIZE) {

            var alpha =
                mask[i]

            if (!alpha.isFinite()) {
                alpha = 0.0f
            }

            /*
             * Support either:
             * 0..1 mask
             * -1..1 mask
             */
            if (alpha < 0.0f) {
                alpha =
                    (alpha + 1.0f) * 0.5f
            }

            alpha =
                alpha.coerceIn(
                    0.0f,
                    1.0f
                )

            val base =
                i * CHANNELS

            for (c in 0 until CHANNELS) {

                val generated =
                    rgb[base + c]
                        .coerceIn(
                            0.0f,
                            1.0f
                        )

                val original =
                    background[base + c]
                        .coerceIn(
                            0.0f,
                            1.0f
                        )

                result[base + c] =
                    generated * alpha +
                    original * (1.0f - alpha)
            }
        }

        return result
    }
}

2. Important: change the result section in "MainActivity.kt"

Find:

val outputBitmap =
    ImageTensor.tensorToBitmap(result.rgb)

Replace it with:

val maskedRgb =
    ImageTensor.applyMask(
        rgb = result.rgb,
        mask = result.mask,
        background = tensor
    )

val outputBitmap =
    ImageTensor.modelRgbToBitmap(
        maskedRgb,
        bgr = false
    )

And keep the diagnostic status:

statusText.text =
    """
    LIAE inference complete

    RGB: ${result.rgb.size}
    Mask: ${result.mask.size}

    RGB min: ${result.rgbMin}
    RGB max: ${result.rgbMax}
    RGB mean: ${result.rgbMean}

    Mask min: ${result.maskMin}
    Mask max: ${result.maskMax}
    Mask mean: ${result.maskMean}
    """.trimIndent()

3. Don't change "LiaeUdEngine.kt" again

Keep the diagnostic version I gave you immediately before this.

Then:

Commit → GitHub Actions → build APK → install → select image.

If the result is still green, send me the six numbers shown on screen. Then we'll determine whether the issue is RGB/BGR ordering, model output semantics, or the ONNX export itself rather than continuing to guess.
