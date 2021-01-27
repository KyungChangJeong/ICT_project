/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.lite.examples.posenet.lib


import android.annotation.SuppressLint
import android.content.Context
import android.content.Context.VIBRATOR_SERVICE
import android.graphics.Bitmap
import android.media.AudioAttributes
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Build
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Toast
import androidx.core.content.ContextCompat.getSystemService

import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.collections.HashMap
import kotlin.math.exp
import org.tensorflow.lite.examples.posenet.*

enum class BodyPart {
    NOSE,
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_EAR,
    RIGHT_EAR,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE
}

var frameCounter = 0
var ActiveCounter = 0

// 실시간 데이터 16가지 각도 체크
var LEFT_SIDE_Arm_angle: Double = 0.0
var LEFT_SIDE_Leg_angle: Double = 0.0
var RIGHT_SIDE_Arm_angle: Double = 0.0
var RIGHT_SIDE_Leg_angle: Double = 0.0
var LEFT_ForeArm_angle: Double = 0.0
var LEFT_Arm_angle: Double = 0.0
var LEFT_Body_angle: Double = 0.0
var LEFT_KneeUp_angle: Double = 0.0
var LEFT_KneeDown_angle: Double = 0.0
var RIGHT_ForeArm_angle: Double = 0.0
var RIGHT_Arm_angle: Double = 0.0
var RIGHT_Body_angle: Double = 0.0
var RIGHT_KneeUp_angle: Double = 0.0
var RIGHT_KneeDown_angle: Double = 0.0
var CENTER_Body_angle: Double = 0.0
var CENTER_Shoulder_angle: Double = 0.0


// Json 데이터 16가지 각도 체크
var JSON_LEFT_SIDE_Arm_angle: Double = 0.0
var JSON_LEFT_SIDE_Leg_angle: Double = 0.0
var JSON_RIGHT_SIDE_Arm_angle: Double = 0.0
var JSON_RIGHT_SIDE_Leg_angle: Double = 0.0
var Json_LEFT_ForeArm_angle: Double = 0.0
var Json_LEFT_Arm_angle: Double = 0.0
var Json_LEFT_Body_angle: Double = 0.0
var Json_LEFT_KneeUp_angle: Double = 0.0
var Json_LEFT_KneeDown_angle: Double = 0.0
var Json_RIGHT_ForeArm_angle: Double = 0.0
var Json_RIGHT_Arm_angle: Double = 0.0
var Json_RIGHT_Body_angle: Double = 0.0
var Json_RIGHT_KneeUp_angle: Double = 0.0
var Json_RIGHT_KneeDown_angle: Double = 0.0
var Json_CENTER_Body_angle: Double = 0.0
var Json_CENTER_Shoulder_angle: Double = 0.0

// 차렷(stand)0 / 왼발(left)1 /차렷(stand)2 /  오른발(right)3
var ActionFlag: Int = 0
var ActionCount: Int = 0
var estimate_LEFT_Arm = ""
var estimate_RIGHT_Arm = ""

var ActionFeedback = ""
var sidejackCount = 0
var ActionScore = 0

var Result_ActionScore = 0

var tts: TextToSpeech? = null

var start = SystemClock.currentThreadTimeMillis();


class Position {
    var x: Int = 0
    var y: Int = 0
}

class KeyPoint {
    var bodyPart: BodyPart = BodyPart.NOSE
    var position: Position = Position()
    var score: Float = 0.0f
}


class Person {
    var keyPoints = listOf<KeyPoint>()
    var score: Float = 0.0f
}

enum class Device {
    CPU,
    NNAPI,
    GPU
}

class Posenet(
    val context: Context,
    val filename: String = "posenet_model.tflite",
    val device: Device = Device.GPU
) : AutoCloseable {
    var lastInferenceTimeNanos: Long = -1
        private set

    /** An Interpreter for the TFLite model.   */
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private val NUM_LITE_THREADS = 4

    private fun getInterpreter(): Interpreter {
        if (interpreter != null) {
            return interpreter!!
        }
        val options = Interpreter.Options()
        options.setNumThreads(NUM_LITE_THREADS)
        when (device) {
            Device.CPU -> {
            }
            Device.GPU -> {
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
            }
            Device.NNAPI -> options.setUseNNAPI(true)
        }
        interpreter = Interpreter(loadModelFile(filename, context), options)
        return interpreter!!
    }

    override fun close() {
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
    }


    /** Returns value within [0,1].   */
    private fun sigmoid(x: Float): Float {
        return (1.0f / (1.0f + exp(-x)))
    }

    /**
     * Scale the image to a byteBuffer of [-1,1] values.
     */
    private fun initInputArray(bitmap: Bitmap): ByteBuffer {
        val bytesPerChannel = 4
        val inputChannels = 3
        val batchSize = 1
        val inputBuffer = ByteBuffer.allocateDirect(
            batchSize * bytesPerChannel * bitmap.height * bitmap.width * inputChannels
        )
        inputBuffer.order(ByteOrder.nativeOrder())
        inputBuffer.rewind()

        val mean = 128.0f
        val std = 128.0f
        val intValues = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (pixelValue in intValues) {
            inputBuffer.putFloat(((pixelValue shr 16 and 0xFF) - mean) / std)
            inputBuffer.putFloat(((pixelValue shr 8 and 0xFF) - mean) / std)
            inputBuffer.putFloat(((pixelValue and 0xFF) - mean) / std)
        }
        return inputBuffer
    }

    /** Preload and memory map the model file, returning a MappedByteBuffer containing the model. */
    private fun loadModelFile(path: String, context: Context): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(path)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        return inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength
        )
    }

    /**
     * Initializes an outputMap of 1 * x * y * z FloatArrays for the model processing to populate.
     */
    private fun initOutputMap(interpreter: Interpreter): HashMap<Int, Any> {
        val outputMap = HashMap<Int, Any>()

        // 1 * 9 * 9 * 17 contains heatmaps
        val heatmapsShape = interpreter.getOutputTensor(0).shape()
        outputMap[0] = Array(heatmapsShape[0]) {
            Array(heatmapsShape[1]) {
                Array(heatmapsShape[2]) { FloatArray(heatmapsShape[3]) }
            }
        }

        // 1 * 9 * 9 * 34 contains offsets
        val offsetsShape = interpreter.getOutputTensor(1).shape()
        outputMap[1] = Array(offsetsShape[0]) {
            Array(offsetsShape[1]) { Array(offsetsShape[2]) { FloatArray(offsetsShape[3]) } }
        }

        // 1 * 9 * 9 * 32 contains forward displacements
        val displacementsFwdShape = interpreter.getOutputTensor(2).shape()
        outputMap[2] = Array(offsetsShape[0]) {
            Array(displacementsFwdShape[1]) {
                Array(displacementsFwdShape[2]) { FloatArray(displacementsFwdShape[3]) }
            }
        }

        // 1 * 9 * 9 * 32 contains backward displacements
        val displacementsBwdShape = interpreter.getOutputTensor(3).shape()
        outputMap[3] = Array(displacementsBwdShape[0]) {
            Array(displacementsBwdShape[1]) {
                Array(displacementsBwdShape[2]) { FloatArray(displacementsBwdShape[3]) }
            }
        }

        return outputMap
    }

    /**
     * Estimates the pose for a single person.
     * args:
     *      bitmap: image bitmap of frame that should be processed
     * returns:
     *      person: a Person object containing data about keypoint locations and confidence scores
     */

    // 사람의 스켈레톤 및 점수 획득
    @Suppress("UNCHECKED_CAST")
    fun estimateSinglePose(bitmap: Bitmap): Person {
        val estimationStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        val inputArray = arrayOf(initInputArray(bitmap))
        Log.i(
            "posenet",
            String.format(
                "Scaling to [-1,1] took %.2f ms",
                1.0f * (SystemClock.elapsedRealtimeNanos() - estimationStartTimeNanos) / 1_000_000
            )
        )

        val outputMap = initOutputMap(getInterpreter())

        val inferenceStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        getInterpreter().runForMultipleInputsOutputs(inputArray, outputMap)
        lastInferenceTimeNanos = SystemClock.elapsedRealtimeNanos() - inferenceStartTimeNanos
        Log.i(
            "posenet",
            String.format("Interpreter took %.2f ms", 1.0f * lastInferenceTimeNanos / 1_000_000)
        )

        val heatmaps = outputMap[0] as Array<Array<Array<FloatArray>>>
        val offsets = outputMap[1] as Array<Array<Array<FloatArray>>>

        val height = heatmaps[0].size
        val width = heatmaps[0][0].size
        val numKeypoints = heatmaps[0][0][0].size

        // Finds the (row, col) locations of where the keypoints are most likely to be.
        val keypointPositions = Array(numKeypoints) { Pair(0, 0) }
        for (keypoint in 0 until numKeypoints) {
            var maxVal = heatmaps[0][0][0][keypoint]
            var maxRow = 0
            var maxCol = 0
            for (row in 0 until height) {
                for (col in 0 until width) {
                    if (heatmaps[0][row][col][keypoint] > maxVal) {
                        maxVal = heatmaps[0][row][col][keypoint]
                        maxRow = row
                        maxCol = col
                    }
                }
            }
            keypointPositions[keypoint] = Pair(maxRow, maxCol)
        }

        // Calculating the x and y coordinates of the keypoints with offset adjustment.
        val xCoords = IntArray(numKeypoints)
        val yCoords = IntArray(numKeypoints)
        val confidenceScores = FloatArray(numKeypoints)
        keypointPositions.forEachIndexed { idx, position ->
            val positionY = keypointPositions[idx].first
            val positionX = keypointPositions[idx].second
            yCoords[idx] = (
                    position.first / (height - 1).toFloat() * bitmap.height +
                            offsets[0][positionY][positionX][idx]
                    ).toInt()
            xCoords[idx] = (
                    position.second / (width - 1).toFloat() * bitmap.width +
                            offsets[0][positionY]
                                    [positionX][idx + numKeypoints]
                    ).toInt()
            confidenceScores[idx] = sigmoid(heatmaps[0][positionY][positionX][idx])
        }

        // 각도 체크
        val pointAngle = 0;

        val person = Person()
        val keypointList = Array(numKeypoints) { KeyPoint() }
        var totalScore = 0.0f
        enumValues<BodyPart>().forEachIndexed { idx, it ->
            keypointList[idx].bodyPart = it
            keypointList[idx].position.x = xCoords[idx]
            keypointList[idx].position.y = yCoords[idx]
            keypointList[idx].score = confidenceScores[idx]
//            Log.d("keypoint.bodyPart", keypointList[idx].bodyPart.toString());
//            Log.d("keypoint.position.x", keypointList[idx].position.x.toString());
//            Log.d("keypoint.position.y", keypointList[idx].position.y.toString());
//            Log.d("keypoint.score", keypointList[idx].score.toString());

            totalScore += confidenceScores[idx]
        }

        //    0. NOSE
        //    1. LEFT_EYE
        //    2. RIGHT_EYE
        //    3. LEFT_EAR
        //    4. RIGHT_EAR
        //    5. LEFT_SHOULDER
        //    6. RIGHT_SHOULDER
        //    7. LEFT_ELBOW
        //    8. RIGHT_ELBOW
        //    9. LEFT_WRIST
        //    10. RIGHT_WRIST
        //    11. LEFT_HIP
        //    12. RIGHT_HIP
        //    13. LEFT_KNEE
        //    14. RIGHT_KNEE
        //    15. LEFT_ANKLE
        //    16. RIGHT_ANKLE





        // 값 수정 X
        person.keyPoints = keypointList.toList()


        // Bodypart 이름(0 ~ 16) / x / y / 신뢰도 출력 가능
//        Log.d("person.keyPoints", person.keyPoints.get(0).bodyPart.toString());
//        Log.d("person.keyPoints", person.keyPoints.get(0).position.x.toString());
//        Log.d("person.keyPoints", person.keyPoints.get(0).position.y.toString());
//        Log.d("person.keyPoints_score", person.keyPoints.get(0).score.toString());

        person.score = totalScore / numKeypoints
        //    Log.d("person.score", person.score.toString());


        // 실시간 데이터 각도 계산

        // LEFT_SIDE_Arm
        var LEFT_SIDE_Arm_dy =
            person.keyPoints.get(9).position.y - person.keyPoints.get(5).position.y
        var LEFT_SIDE_Arm_dx =
            person.keyPoints.get(9).position.x - person.keyPoints.get(5).position.x
        LEFT_SIDE_Arm_angle =
            Math.atan2(LEFT_SIDE_Arm_dy.toDouble(), LEFT_SIDE_Arm_dx.toDouble()) * (180.0 / Math.PI)
//        Log.d("LEFT_SIDE_Arm", LEFT_SIDE_Arm_angle.toString());

        // LEFT_SIDE_Leg
        var LEFT_SIDE_Leg_dy =
            person.keyPoints.get(15).position.y - person.keyPoints.get(11).position.y
        var LEFT_SIDE_Leg_dx =
            person.keyPoints.get(15).position.x - person.keyPoints.get(11).position.x
        LEFT_SIDE_Leg_angle =
            Math.atan2(LEFT_SIDE_Leg_dy.toDouble(), LEFT_SIDE_Leg_dx.toDouble()) * (180.0 / Math.PI)
//        Log.d("LEFT_SIDE_Leg", LEFT_SIDE_Leg_angle.toString());

        // RIGHT_SIDE_Arm
        var RIGHT_SIDE_Arm_dy =
            person.keyPoints.get(6).position.y - person.keyPoints.get(10).position.y
        var RIGHT_SIDE_Arm_dx =
            person.keyPoints.get(6).position.x - person.keyPoints.get(10).position.x
        RIGHT_SIDE_Arm_angle =
            Math.atan2(
                RIGHT_SIDE_Arm_dy.toDouble(),
                RIGHT_SIDE_Arm_dx.toDouble()
            ) * (180.0 / Math.PI)
//        Log.d("RIGHT_SIDE_Arm", RIGHT_SIDE_Arm_angle.toString());

        // RIGHT_SIDE_Leg
        var RIGHT_SIDE_Leg_dy =
            person.keyPoints.get(16).position.y - person.keyPoints.get(12).position.y
        var RIGHT_SIDE_Leg_dx =
            person.keyPoints.get(16).position.x - person.keyPoints.get(12).position.x
        RIGHT_SIDE_Leg_angle =
            Math.atan2(
                RIGHT_SIDE_Leg_dy.toDouble(),
                RIGHT_SIDE_Leg_dx.toDouble()
            ) * (180.0 / Math.PI)
//        Log.d("RIGHT_SIDE_Leg", RIGHT_SIDE_Leg_angle.toString());


        // LEFT_ForeArm
        var LEFT_ForeArm_dy =
            person.keyPoints.get(9).position.y - person.keyPoints.get(7).position.y
        var LEFT_ForeArm_dx =
            person.keyPoints.get(9).position.x - person.keyPoints.get(7).position.x
        LEFT_ForeArm_angle =
            Math.atan2(LEFT_ForeArm_dy.toDouble(), LEFT_ForeArm_dx.toDouble()) * (180.0 / Math.PI)
//        Log.d("LEFT_ForeArm", LEFT_ForeArm_angle.toString());

        // LEFT_Arm
        var LEFT_Arm_dy = person.keyPoints.get(7).position.y - person.keyPoints.get(5).position.y
        var LEFT_Arm_dx = person.keyPoints.get(7).position.x - person.keyPoints.get(5).position.x
        LEFT_Arm_angle =
            Math.atan2(LEFT_Arm_dy.toDouble(), LEFT_Arm_dx.toDouble()) * (180.0 / Math.PI)
//        Log.d("LEFT_Arm", LEFT_Arm_angle.toString());

        // LEFT_Body
        var LEFT_Body_dy = person.keyPoints.get(5).position.y - person.keyPoints.get(11).position.y
        var LEFT_Body_dx = person.keyPoints.get(5).position.x - person.keyPoints.get(11).position.x
        LEFT_Body_angle =
            Math.atan2(LEFT_Body_dy.toDouble(), LEFT_Body_dx.toDouble()) * (180.0 / Math.PI)
//        Log.d("LEFT_Body", LEFT_Body_angle.toString());

        // LEFT_KneeUp
        var LEFT_KneeUp_dy =
            person.keyPoints.get(11).position.y - person.keyPoints.get(13).position.y
        var LEFT_KneeUp_dx =
            person.keyPoints.get(11).position.x - person.keyPoints.get(13).position.x
        LEFT_KneeUp_angle =
            Math.atan2(LEFT_KneeUp_dy.toDouble(), LEFT_KneeUp_dx.toDouble()) * (180.0 / Math.PI)
//        Log.d("LEFT_KneeUp", LEFT_KneeUp_angle.toString());

        // LEFT_KneeDown
        var LEFT_KneeDown_dy =
            person.keyPoints.get(13).position.y - person.keyPoints.get(15).position.y
        var LEFT_KneeDown_dx =
            person.keyPoints.get(13).position.x - person.keyPoints.get(15).position.x
        LEFT_KneeDown_angle =
            Math.atan2(LEFT_KneeDown_dy.toDouble(), LEFT_KneeDown_dx.toDouble()) * (180.0 / Math.PI)
//        Log.d("LEFT_KneeDown", LEFT_KneeDown_angle.toString());

        // RIGHT_ForeArm
        var RIGHT_ForeArm_dy =
            person.keyPoints.get(10).position.y - person.keyPoints.get(8).position.y
        var RIGHT_ForeArm_dx =
            person.keyPoints.get(10).position.x - person.keyPoints.get(8).position.x
        RIGHT_ForeArm_angle =
            Math.atan2(RIGHT_ForeArm_dy.toDouble(), RIGHT_ForeArm_dx.toDouble()) * (180.0 / Math.PI)
//        Log.d("RIGHT_ForeArm", RIGHT_ForeArm_angle.toString());

        // RIGHT_Arm
        var RIGHT_Arm_dy = person.keyPoints.get(8).position.y - person.keyPoints.get(6).position.y
        var RIGHT_Arm_dx = person.keyPoints.get(8).position.x - person.keyPoints.get(6).position.x
        RIGHT_Arm_angle =
            Math.atan2(RIGHT_Arm_dy.toDouble(), RIGHT_Arm_dx.toDouble()) * (180.0 / Math.PI)
//        Log.d("RIGHT_Arm", RIGHT_Arm_angle.toString());

        // RIGHT_Body
        var RIGHT_Body_dy = person.keyPoints.get(6).position.y - person.keyPoints.get(12).position.y
        var RIGHT_Body_dx = person.keyPoints.get(6).position.x - person.keyPoints.get(12).position.x
        RIGHT_Body_angle =
            Math.atan2(RIGHT_Body_dy.toDouble(), RIGHT_Body_dx.toDouble()) * (180.0 / Math.PI)
//        Log.d("RIGHT_Body", RIGHT_Body_angle.toString());

        // RIGHT_KneeUp
        var RIGHT_KneeUp_dy =
            person.keyPoints.get(12).position.y - person.keyPoints.get(14).position.y
        var RIGHT_KneeUp_dx =
            person.keyPoints.get(12).position.x - person.keyPoints.get(14).position.x
        RIGHT_KneeUp_angle =
            Math.atan2(RIGHT_KneeUp_dy.toDouble(), RIGHT_KneeUp_dx.toDouble()) * (180.0 / Math.PI)
//        Log.d("RIGHT_KneeUp", RIGHT_KneeUp_angle.toString());

        // RIGHT_KneeDown
        var RIGHT_KneeDown_dy =
            person.keyPoints.get(14).position.y - person.keyPoints.get(16).position.y
        var RIGHT_KneeDown_dx =
            person.keyPoints.get(14).position.x - person.keyPoints.get(16).position.x
        RIGHT_KneeDown_angle = Math.atan2(
            RIGHT_KneeDown_dy.toDouble(),
            RIGHT_KneeDown_dx.toDouble()
        ) * (180.0 / Math.PI)
//        Log.d("RIGHT_KneeDown", RIGHT_KneeDown_angle.toString());

        // CENTER_Body
        var CENTER_Body_dy = person.keyPoints.get(5).position.y - person.keyPoints.get(6).position.y
        var CENTER_Body_dx = person.keyPoints.get(5).position.x - person.keyPoints.get(6).position.x
        CENTER_Body_angle =
            Math.atan2(CENTER_Body_dy.toDouble(), CENTER_Body_dx.toDouble()) * (180.0 / Math.PI)
//        Log.d("CENTER_Body", CENTER_Body_angle.toString());

        // CENTER_Shoulder
        var CENTER_Shoulder_dy =
            person.keyPoints.get(11).position.y - person.keyPoints.get(12).position.y
        var CENTER_Shoulder_dx =
            person.keyPoints.get(11).position.x - person.keyPoints.get(12).position.x
        CENTER_Shoulder_angle = Math.atan2(
            CENTER_Shoulder_dy.toDouble(),
            CENTER_Shoulder_dx.toDouble()
        ) * (180.0 / Math.PI)
//        Log.d("CENTER_Shoulder", CENTER_Shoulder_angle.toString());




//        //tts 생성
//
//
//        val tts = TextToSpeech(this.context) {
//            if (it == TextToSpeech.SUCCESS) {
//                val result = tts?.setLanguage(Locale.KOREAN)
//                // 언어 설정
//            }
//        }
//
//        tts.setSpeechRate(0.9f)
//        // 말하는 속도 설정
//
//        tts.speak("말할 텍스트를 입력!", TextToSpeech.QUEUE_FLUSH, null, null)
//        // 말해!
//
//        tts.stop()
//        // tts speaking 중지
//
//        tts.shutdown()
//        // tts 중지

        // 프레임별 실시간데이터 & 선생데이터 비교

        // 실시간 데이터 점수가 낮을경우
        // 한 프레임 비교 안하도록
//        Log.d("totalScore", totalScore.toString())
//        if (totalScore >= 90) {
//            // 10Frame 별로 실시간데이터 평가
//
//        }
//        else{
//
//        }


        // 선생데이터 (JSON파일 프레임 데이터 추출)
        jsonObjectsExample()



        // 학습
        poseEstimate(person);

        // 운동
        FrameComparison();




//        if((frameCounter %10) == 0) {
//
//            var frame_end = SystemClock.currentThreadTimeMillis();
//            Log.d("운동 한세트 걸린시간 : ",(frame_end - frame_start).toString());
//            frame_start = frame_end;
//
//        }

        // 사람 점수가 잘 나오면 평가 시작
        if (person.score >= 80){

        }
        else {

        }

        frameCounter++;

        return person
    }


    // Json data (선생 데이터) 가져오기
    @SuppressLint("LongLogTag")
    fun jsonObjectsExample() {
        // 파일 경로 세팅 완료
        var filePathFirst = "sidejack/"
        var filePathFinal = ".json"

        // 실제
        var fileJsonPath = filePathFirst+ ActiveCounter + filePathFinal
//        Log.d("fileJsonPath :",fileJsonPath)
        // test용
//        var fileJsonPath = filePathFirst + 3 + filePathFinal

//        Log.d("JSON_FRAME_COUNTER", ActiveCounter.toString());
//        Log.d("파일 경로 확인", fileJsonPath)

        // open 해결 => 0 ~ 268 Frame의 정보 가져오도록

        val inputStream = context.assets.open("$fileJsonPath")
        val br = BufferedReader(InputStreamReader(inputStream))
        val str = br.readText()
        val jo = JSONObject(str)

//        Log.d("jo :", jo.toString())

        // 객체 불러옴
        val jArray = jo.getJSONArray("Frame")
//        val jArray = jo.getJSONArray("FRAME$ActiveCounter")

        for (i in 0 until jArray.length()) {
            val obj = jArray.getJSONObject(i)
            val json_bodypart = obj.getString("NAME")
            val json_x = obj.getDouble("x")
            val json_y = obj.getDouble("y")
            val json_score = obj.getDouble("score")

//            Log.d("JSON_DATA", "NAME($i): $json_bodypart")
//            Log.d("JSON_DATA", "x($i): $json_x")
//            Log.d("JSON_DATA", "y($i): $json_y")
//            Log.d("JSON_DATA", "y($i): $json_score")
        }




        // JSON_LEFT_SIDE_Arm
        val JSON_LEFT_SIDE_Arm_X =
            jArray.getJSONObject(9).getInt("x") - jArray.getJSONObject(5).getInt(
                "x"
            )
        val JSON_LEFT_SIDE_Arm_Y =
            jArray.getJSONObject(9).getInt("y") - jArray.getJSONObject(5).getInt(
                "y"
            )
        JSON_LEFT_SIDE_Arm_angle = Math.atan2(
            JSON_LEFT_SIDE_Arm_Y.toDouble(),
            JSON_LEFT_SIDE_Arm_X.toDouble()
        ) * (180.0 / Math.PI)
//        Log.d("JSON_LEFT_SIDE_Arm_angle", JSON_LEFT_SIDE_Arm_angle.toString());

        // JSON_LEFT_SIDE_Leg
        val JSON_LEFT_SIDE_Leg_X =
            jArray.getJSONObject(15).getInt("x") - jArray.getJSONObject(11).getInt(
                "x"
            )
        val JSON_LEFT_SIDE_Leg_Y =
            jArray.getJSONObject(15).getInt("y") - jArray.getJSONObject(11).getInt(
                "y"
            )
        JSON_LEFT_SIDE_Leg_angle = Math.atan2(
            JSON_LEFT_SIDE_Leg_Y.toDouble(),
            JSON_LEFT_SIDE_Leg_X.toDouble()
        ) * (180.0 / Math.PI)
//        Log.d("JSON_LEFT_SIDE_Leg_angle", JSON_LEFT_SIDE_Leg_angle.toString());

        // JSON_RIGHT_SIDE_Arm
        val JSON_RIGHT_SIDE_Arm_X =
            jArray.getJSONObject(6).getInt("x") - jArray.getJSONObject(10).getInt(
                "x"
            )
        val JSON_RIGHT_SIDE_Arm_Y =
            jArray.getJSONObject(6).getInt("y") - jArray.getJSONObject(10).getInt(
                "y"
            )
        JSON_RIGHT_SIDE_Arm_angle = Math.atan2(
            JSON_RIGHT_SIDE_Arm_Y.toDouble(),
            JSON_RIGHT_SIDE_Arm_X.toDouble()
        ) * (180.0 / Math.PI)
//        Log.d("JSON_RIGHT_SIDE_Arm_angle", JSON_RIGHT_SIDE_Arm_angle.toString());

        // JSON_RIGHT_SIDE_Leg
        val JSON_RIGHT_SIDE_Leg_X =
            jArray.getJSONObject(16).getInt("x") - jArray.getJSONObject(12).getInt(
                "x"
            )
        val JSON_RIGHT_SIDE_Leg_Y =
            jArray.getJSONObject(16).getInt("y") - jArray.getJSONObject(12).getInt(
                "y"
            )
        JSON_RIGHT_SIDE_Leg_angle = Math.atan2(
            JSON_RIGHT_SIDE_Leg_Y.toDouble(),
            JSON_RIGHT_SIDE_Leg_X.toDouble()
        ) * (180.0 / Math.PI)
//        Log.d("JSON_RIGHT_SIDE_Leg_angle", JSON_RIGHT_SIDE_Leg_angle.toString());

        // Json_Left_ForeArm
        val Json_LEFT_ForeArm_X =
            jArray.getJSONObject(9).getInt("x") - jArray.getJSONObject(7).getInt(
                "x"
            )
//        Log.d("Json_Left_ForeArm_X", Json_LEFT_ForeArm_X.toString());
        val Json_LEFT_ForeArm_Y =
            jArray.getJSONObject(9).getInt("y") - jArray.getJSONObject(7).getInt(
                "y"
            )
//        Log.d("Json_Left_ForeArm_Y", Json_LEFT_ForeArm_Y.toString());
        Json_LEFT_ForeArm_angle = Math.atan2(
            Json_LEFT_ForeArm_Y.toDouble(),
            Json_LEFT_ForeArm_X.toDouble()
        ) * (180.0 / Math.PI)
//        Log.d("Json_Left_ForeArm_angle", Json_LEFT_ForeArm_angle.toString());

        // Json_LEFT_Arm
        val Json_LEFT_Arm_X =
            jArray.getJSONObject(7).getInt("x") - jArray.getJSONObject(5).getInt("x")
//        Log.d("Json_LEFT_Arm_X", Json_LEFT_Arm_X.toString());
        val Json_LEFT_Arm_Y =
            jArray.getJSONObject(7).getInt("y") - jArray.getJSONObject(5).getInt("y")
//        Log.d("Json_LEFT_Arm_Y", Json_LEFT_Arm_Y.toString());
        Json_LEFT_Arm_angle =
            Math.atan2(Json_LEFT_Arm_Y.toDouble(), Json_LEFT_Arm_X.toDouble()) * (180.0 / Math.PI)
//        Log.d("Json_LEFT_Arm_angle", Json_LEFT_Arm_angle.toString());

        // Json_LEFT_Body
        val Json_LEFT_Body_X =
            jArray.getJSONObject(5).getInt("x") - jArray.getJSONObject(11).getInt(
                "x"
            )
//        Log.d("Json_LEFT_Body_X", Json_LEFT_Body_X.toString());
        val Json_LEFT_Body_Y =
            jArray.getJSONObject(5).getInt("y") - jArray.getJSONObject(11).getInt(
                "y"
            )
//        Log.d("Json_LEFT_Body_Y", Json_LEFT_Body_Y.toString());
        Json_LEFT_Body_angle = Math.atan2(
            Json_LEFT_Body_Y.toDouble(),
            Json_LEFT_Body_X.toDouble()
        ) * (180.0 / Math.PI)
//        Log.d("Json_LEFT_Body_angle", Json_LEFT_Body_angle.toString());

        // Json_LEFT_KneeUp
        val Json_LEFT_KneeUp_X =
            jArray.getJSONObject(11).getInt("x") - jArray.getJSONObject(13).getInt(
                "x"
            )
//        Log.d("Json_LEFT_KneeUp_X", Json_LEFT_KneeUp_X.toString());
        val Json_LEFT_KneeUp_Y =
            jArray.getJSONObject(11).getInt("y") - jArray.getJSONObject(13).getInt(
                "y"
            )
//        Log.d("Json_LEFT_KneeUp_Y", Json_LEFT_KneeUp_Y.toString());
        Json_LEFT_KneeUp_angle = Math.atan2(
            Json_LEFT_KneeUp_Y.toDouble(),
            Json_LEFT_KneeUp_X.toDouble()
        ) * (180.0 / Math.PI)
//        Log.d("Json_LEFT_KneeUp_angle", Json_LEFT_KneeUp_angle.toString());

        // Json_LEFT_KneeDown
        val Json_LEFT_KneeDown_X =
            jArray.getJSONObject(13).getInt("x") - jArray.getJSONObject(15).getInt(
                "x"
            )
//        Log.d("Json_LEFT_KneeDown_X", Json_LEFT_KneeDown_X.toString());
        val Json_LEFT_KneeDown_Y =
            jArray.getJSONObject(13).getInt("y") - jArray.getJSONObject(15).getInt(
                "y"
            )
//        Log.d("Json_LEFT_KneeDown_Y", Json_LEFT_KneeDown_Y.toString());
        Json_LEFT_KneeDown_angle = Math.atan2(
            Json_LEFT_KneeDown_Y.toDouble(),
            Json_LEFT_KneeDown_X.toDouble()
        ) * (180.0 / Math.PI)
//        Log.d("Json_LEFT_KneeDown_angle", Json_LEFT_KneeDown_angle.toString());


        // Json_RIGHT_ForeArm
        val Json_RIGHT_ForeArm_X =
            jArray.getJSONObject(10).getInt("x") - jArray.getJSONObject(8).getInt(
                "x"
            )
//        Log.d("Json_RIGHT_ForeArm_X", Json_RIGHT_ForeArm_X.toString());
        val Json_RIGHT_ForeArm_Y =
            jArray.getJSONObject(10).getInt("y") - jArray.getJSONObject(8).getInt(
                "y"
            )
//        Log.d("Json_RIGHT_ForeArm_Y", Json_LEFT_ForeArm_Y.toString());
        Json_RIGHT_ForeArm_angle = Math.atan2(
            Json_RIGHT_ForeArm_Y.toDouble(),
            Json_RIGHT_ForeArm_X.toDouble()
        ) * (180.0 / Math.PI)
//        Log.d("Json_RIGHT_ForeArm_angle", Json_RIGHT_ForeArm_angle.toString());

        // Json_RIGHT_Arm
        val Json_RIGHT_Arm_X = jArray.getJSONObject(8).getInt("x") - jArray.getJSONObject(6).getInt(
            "x"
        )
//        Log.d("Json_RIGHT_Arm_X", Json_RIGHT_Arm_X.toString());
        val Json_RIGHT_Arm_Y = jArray.getJSONObject(8).getInt("y") - jArray.getJSONObject(6).getInt(
            "y"
        )
//        Log.d("Json_RIGHT_Arm_Y", Json_RIGHT_Arm_Y.toString());
        Json_RIGHT_Arm_angle = Math.atan2(
            Json_RIGHT_Arm_Y.toDouble(),
            Json_RIGHT_Arm_X.toDouble()
        ) * (180.0 / Math.PI)
//        Log.d("Json_RIGHT_Arm_angle", Json_RIGHT_Arm_angle.toString());

        // Json_RIGHT_Body
        val Json_RIGHT_Body_X =
            jArray.getJSONObject(6).getInt("x") - jArray.getJSONObject(12).getInt(
                "x"
            )
//        Log.d("Json_RIGHT_Body_X", Json_RIGHT_Body_X.toString());
        val Json_RIGHT_Body_Y =
            jArray.getJSONObject(6).getInt("y") - jArray.getJSONObject(12).getInt(
                "y"
            )
//        Log.d("Json_RIGHT_Body_Y", Json_RIGHT_Body_Y.toString());
        Json_RIGHT_Body_angle = Math.atan2(
            Json_RIGHT_Body_Y.toDouble(),
            Json_RIGHT_Body_X.toDouble()
        ) * (180.0 / Math.PI)
//        Log.d("Json_RIGHT_Body_angle", Json_RIGHT_Body_angle.toString());

        // Json_RIGHT_KneeUp
        val Json_RIGHT_KneeUp_X =
            jArray.getJSONObject(12).getInt("x") - jArray.getJSONObject(14).getInt(
                "x"
            )
//        Log.d("Json_RIGHT_KneeUp_X", Json_RIGHT_KneeUp_X.toString());
        val Json_RIGHT_KneeUp_Y =
            jArray.getJSONObject(12).getInt("y") - jArray.getJSONObject(14).getInt(
                "y"
            )
//        Log.d("Json_RIGHT_KneeUp_Y", Json_RIGHT_KneeUp_Y.toString());
        Json_RIGHT_KneeUp_angle = Math.atan2(
            Json_RIGHT_KneeUp_Y.toDouble(),
            Json_RIGHT_KneeUp_X.toDouble()
        ) * (180.0 / Math.PI)
//        Log.d("Json_RIGHT_KneeUp_angle", Json_RIGHT_KneeUp_angle.toString());

        // Json_RIGHT_KneeDown
        val Json_RIGHT_KneeDown_X =
            jArray.getJSONObject(14).getInt("x") - jArray.getJSONObject(16).getInt(
                "x"
            )
//        Log.d("Json_RIGHT_KneeDown_X", Json_RIGHT_KneeDown_X.toString());
        val Json_RIGHT_KneeDown_Y =
            jArray.getJSONObject(14).getInt("y") - jArray.getJSONObject(16).getInt(
                "y"
            )
//        Log.d("Json_RIGHT_KneeDown_Y", Json_RIGHT_KneeDown_Y.toString());
        Json_RIGHT_KneeDown_angle = Math.atan2(
            Json_RIGHT_KneeDown_Y.toDouble(),
            Json_RIGHT_KneeDown_X.toDouble()
        ) * (180.0 / Math.PI)
//        Log.d("Json_RIGHT_KneeDown_angle", Json_RIGHT_KneeDown_angle.toString());

        // Json_CENTER_Body
        val Json_CENTER_Body_X =
            jArray.getJSONObject(5).getInt("x") - jArray.getJSONObject(6).getInt(
                "x"
            )
//        Log.d("Json_CENTER_Body_X", Json_CENTER_Body_X.toString());
        val Json_CENTER_Body_Y =
            jArray.getJSONObject(5).getInt("y") - jArray.getJSONObject(6).getInt(
                "y"
            )
//        Log.d("Json_CENTER_Body_Y", Json_CENTER_Body_Y.toString());
        Json_CENTER_Body_angle = Math.atan2(
            Json_CENTER_Body_Y.toDouble(),
            Json_CENTER_Body_X.toDouble()
        ) * (180.0 / Math.PI)
//        Log.d("Json_CENTER_Body_angle", Json_CENTER_Body_angle.toString());

        // Json_CENTER_Shoulder
        val Json_CENTER_Shoulder_X =
            jArray.getJSONObject(11).getInt("x") - jArray.getJSONObject(12).getInt(
                "x"
            )
//        Log.d("Json_CENTER_Shoulder_X", Json_CENTER_Shoulder_X.toString());
        val Json_CENTER_Shoulder_Y =
            jArray.getJSONObject(11).getInt("y") - jArray.getJSONObject(12).getInt(
                "y"
            )
//        Log.d("Json_CENTER_Shoulder_Y", Json_CENTER_Shoulder_Y.toString());
        Json_CENTER_Shoulder_angle = Math.atan2(
            Json_CENTER_Shoulder_Y.toDouble(),
            Json_CENTER_Shoulder_X.toDouble()
        ) * (180.0 / Math.PI)
//        Log.d("Json_CENTER_Shoulder_angle", Json_CENTER_Shoulder_angle.toString());

//        if(ActiveCounter == 130|| ActiveCounter == 149|| ActiveCounter == 168){
//            val toneGen1 = ToneGenerator(AudioManager.STREAM_MUSIC, 200)
//            toneGen1.startTone(ToneGenerator.TONE_CDMA_ABBR_ALERT, 300)
//        }

        if(ActiveCounter == 150){
            val toneGen1 = ToneGenerator(AudioManager.STREAM_MUSIC, 200)
            toneGen1.startTone(ToneGenerator.TONE_CDMA_ABBR_ALERT, 100)
            var end = SystemClock.currentThreadTimeMillis();
            Log.d("운동 한세트 걸린시간 : ",(end - start).toString());
            start = end;

            ActiveCounter = 0
            sidejackCount++;
            Log.d("ActiveCounter", ActiveCounter.toString() )
        }else{
            ActiveCounter++;
            Log.d("ActiveCounter", ActiveCounter.toString() )
        }

    }

    // 운동 프레임 비교
    fun FrameComparison(){

        // 바운드 높게 줄수록 점수 높음
        if(Math.abs(LEFT_SIDE_Arm_angle - JSON_LEFT_SIDE_Arm_angle) <= 10 || Math.abs(RIGHT_SIDE_Arm_angle - JSON_RIGHT_SIDE_Arm_angle) <= 10){
            ActionScore += 100;
        }
        else if(Math.abs(LEFT_SIDE_Arm_angle - JSON_LEFT_SIDE_Arm_angle) <= 25 || Math.abs(RIGHT_SIDE_Arm_angle - JSON_RIGHT_SIDE_Arm_angle) <= 25){
            ActionScore += 90;
        }
        else if(Math.abs(LEFT_SIDE_Arm_angle - JSON_LEFT_SIDE_Arm_angle) <= 40 || Math.abs(RIGHT_SIDE_Arm_angle - JSON_RIGHT_SIDE_Arm_angle) <= 40){
            ActionScore += 80;
        }
//        else if(Math.abs(LEFT_SIDE_Arm_angle - JSON_LEFT_SIDE_Arm_angle) <= 30 || Math.abs(RIGHT_SIDE_Arm_angle - JSON_RIGHT_SIDE_Arm_angle) <= 30){
//            ActionScore += 70;
//        }
        else {
            ActionScore += 50;
        }

//        if(Math.abs(RIGHT_SIDE_Arm_angle - JSON_RIGHT_SIDE_Arm_angle) <= 15){
//            ActionScore += 100;
//        }
//        else if(Math.abs(RIGHT_SIDE_Arm_angle - JSON_RIGHT_SIDE_Arm_angle) <= 25){
//            ActionScore += 90;
//        }
//        else if(Math.abs(RIGHT_SIDE_Arm_angle - JSON_RIGHT_SIDE_Arm_angle) <= 35){
//            ActionScore += 80;
//        }
//        else{
//            ActionScore += 60;
//        }

        Log.d("팔 데이터 값 비교 : ", (LEFT_SIDE_Arm_angle - JSON_LEFT_SIDE_Arm_angle).toString()  )
        Log.d("팔 데이터 값 비교 : ", (RIGHT_SIDE_Arm_angle - JSON_RIGHT_SIDE_Arm_angle).toString()  )



        if ((frameCounter % 15) == 0)  {

            Log.d("ActionScore : ", ActionScore.toString())
            Result_ActionScore  = ActionScore / 15
            Log.d("Result_ActionScore : ", Result_ActionScore.toString())

            if((Result_ActionScore) >= 90 ){
                Log.d("평가중 굳 ActionScore : ", (Result_ActionScore).toString())
                ActionFeedback = "Good"
                Log.d("ActionFeedback : ",ActionFeedback)
                ActionScore = 0
                Result_ActionScore = 0
            }
            else if((Result_ActionScore) >= 70 ){
                Log.d("평가중 노말 ActionScore : ", (Result_ActionScore).toString())
                ActionFeedback = "Normal"
                Log.d("ActionFeedback : ",ActionFeedback)
                ActionScore = 0
                Result_ActionScore = 0
            }
            else {
                Log.d("평가중 뱃 ActionScore : ", (Result_ActionScore).toString())
                ActionFeedback = "Bad"
                Log.d("ActionFeedback : ",ActionFeedback)
                ActionScore = 0
                Result_ActionScore = 0
            }
        }
    }



    // 튜토리얼로 활용
    fun poseEstimate(person: Person) {
        // 자세 평가
        var Estimate_Arm_Bound =
            person.keyPoints.get(9).score + person.keyPoints.get(7).score + person.keyPoints.get(5).score + person.keyPoints.get(
                6
            ).score + person.keyPoints.get(8).score + person.keyPoints.get(10).score;


        var Estimate_Leg_Bound =
            person.keyPoints.get(15).score + person.keyPoints.get(13).score + person.keyPoints.get(
                11
            ).score + person.keyPoints.get(16).score + person.keyPoints.get(14).score + person.keyPoints.get(
                12
            ).score;


        Log.d("Estimate_Arm_Bound : ", Estimate_Arm_Bound.toString());

        // 값 초기화
        estimate_LEFT_Arm = ""
        estimate_RIGHT_Arm = ""

        // 부위별 score가 0.8 ~ 0.9이면 프레임 평가
        if (Estimate_Arm_Bound > 5) {
            // 동작 상태에 따라 다르게
            if (ActionFlag == 0 || ActionFlag == 2) {
                // 팔
                // 차렷(stand)0 / 왼발(left)1 /차렷(stand)2 /  오른발(right)39
                if (LEFT_SIDE_Arm_angle <= 110 && LEFT_SIDE_Arm_angle >= 70) {
                    estimate_LEFT_Arm = "Good"
                    Log.d("차렷 왼팔 : ", estimate_LEFT_Arm);
                } else {
                    estimate_LEFT_Arm = "왼팔을 몸쪽으로"
                    Log.d("왼팔을 몸쪽으로 : ", estimate_LEFT_Arm);
                }
                if (RIGHT_SIDE_Arm_angle <= -70 && RIGHT_SIDE_Arm_angle >= -110) {
                    estimate_RIGHT_Arm = "Good"
                    Log.d("차렷 오른팔 : ", estimate_RIGHT_Arm);
                } else {
                    estimate_RIGHT_Arm = "오른팔을 몸쪽으로"
                    Log.d("오른팔을 몸쪽으로 : ", estimate_RIGHT_Arm);
                }


                if (ActionFlag == 0 && (estimate_LEFT_Arm == "Good" && estimate_RIGHT_Arm == "Good")) {
                    ActionFlag = 1;
                } else if (ActionFlag == 2 && (estimate_LEFT_Arm == "Good" && estimate_RIGHT_Arm == "Good")) {
                    ActionFlag = 3;
                }


            } else if (ActionFlag == 1) {
                // 팔
                if (LEFT_SIDE_Arm_angle <= 10 && LEFT_SIDE_Arm_angle >= -10) {
                    estimate_LEFT_Arm = "Good"
                    Log.d("쭉 핀 왼팔 : ", estimate_LEFT_Arm);
                } else if (LEFT_SIDE_Arm_angle <= 60 && LEFT_SIDE_Arm_angle >= 30) {
                    estimate_LEFT_Arm = "왼팔을 높게"
                    Log.d("낮게 올린 왼팔 : ", estimate_LEFT_Arm);
                } else if (LEFT_SIDE_Arm_angle < 30 && LEFT_SIDE_Arm_angle >= -45) {
                    estimate_LEFT_Arm = "왼팔을 낮게"
                    Log.d("많이 올린 왼팔 : ", estimate_LEFT_Arm);
                }
                if (RIGHT_SIDE_Arm_angle <= 10 && RIGHT_SIDE_Arm_angle >= -10) {
                    estimate_RIGHT_Arm = "Good"
                    Log.d("쭉 핀 오른팔 : ", estimate_RIGHT_Arm);
                } else if (RIGHT_SIDE_Arm_angle <= 60 && RIGHT_SIDE_Arm_angle >= 30) {
                    estimate_RIGHT_Arm = "오른팔을 높게"
                    Log.d("낮게 올린 오른팔 : ", estimate_RIGHT_Arm);
                } else if (RIGHT_SIDE_Arm_angle < 30 && RIGHT_SIDE_Arm_angle >= -45) {
                    estimate_RIGHT_Arm = "오른팔을 낮게"
                    Log.d("많이 올린 오른팔 : ", estimate_RIGHT_Arm);
                }
                // 다리 평가?
                if (ActionFlag == 1 && (estimate_LEFT_Arm == "Good" && estimate_RIGHT_Arm == "Good")) {
                    ActionFlag = 2
                }

            } else if (ActionFlag == 3) {
                // 팔
                if (LEFT_SIDE_Arm_angle <= 10 && LEFT_SIDE_Arm_angle >= -10) {
                    estimate_LEFT_Arm = "Good"
                    Log.d("쭉 핀 왼팔 : ", estimate_LEFT_Arm);
                } else if (LEFT_SIDE_Arm_angle <= 60 && LEFT_SIDE_Arm_angle >= 30) {
                    estimate_LEFT_Arm = "왼팔을 높게"
                    Log.d("낮게 올린 왼팔 : ", estimate_LEFT_Arm);
                } else if (LEFT_SIDE_Arm_angle < 30 && LEFT_SIDE_Arm_angle >= -45) {
                    estimate_LEFT_Arm = "왼팔을 낮게"
                    Log.d("많이 올린 왼팔 : ", estimate_LEFT_Arm);
                }
                if (RIGHT_SIDE_Arm_angle <= 10 && RIGHT_SIDE_Arm_angle >= -10) {
                    estimate_RIGHT_Arm = "Good"
                    Log.d("쭉 핀 오른팔 : ", estimate_RIGHT_Arm);
                } else if (RIGHT_SIDE_Arm_angle <= 60 && RIGHT_SIDE_Arm_angle >= 30) {
                    estimate_RIGHT_Arm = "오른팔을 높게"
                    Log.d("낮게 올린 오른팔 : ", estimate_RIGHT_Arm);
                } else if (RIGHT_SIDE_Arm_angle < 30 && RIGHT_SIDE_Arm_angle >= -45) {
                    estimate_RIGHT_Arm = "오른팔을 낮게"
                    Log.d("많이 올린 오른팔 : ", estimate_RIGHT_Arm);
                }
                // 다리 평가?
                if (ActionFlag == 3 && (estimate_LEFT_Arm == "Good" && estimate_RIGHT_Arm == "Good")) {
                    ActionFlag = 0
                    ActionCount++
                }
            }
        } else {
            Log.d("사용자 데이터가 옳바르지 않아 평가 X", Estimate_Arm_Bound.toString());
//            estimate_LEFT_Arm = "BAD"
//            estimate_RIGHT_Arm = "BAD"
        }


        // 다리
        // 부위별 score가 0.8 ~ 0.9
        if (Estimate_Leg_Bound > 5) {

        }
    }
}