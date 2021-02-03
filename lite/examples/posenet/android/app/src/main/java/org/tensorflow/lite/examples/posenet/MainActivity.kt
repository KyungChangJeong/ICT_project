package org.tensorflow.lite.examples.posenet

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.LinearLayout

var ClickState = ""

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val tutorial = findViewById<LinearLayout>(R.id.tutorial)
        val sidejack = findViewById<LinearLayout>(R.id.sidejack)
        val widesquat = findViewById<LinearLayout>(R.id.widesquat)

        tutorial.setOnClickListener({
            val intent = Intent(this, CameraActivity::class.java)
            ClickState = "sidejack 학습";
            Log.d("sidejack 학습", ClickState)
            startActivity(intent)
        })

        sidejack.setOnClickListener({
            val intent = Intent(this, CameraActivity::class.java)
            ClickState = "sidejack 운동";
            Log.d("sidejack 운동", ClickState)
            startActivity(intent)
        })

        widesquat.setOnClickListener({
            val intent = Intent(this, CameraActivity::class.java)
            ClickState = "widesquat";
            Log.d("widesquat", ClickState)
            startActivity(intent)
        })


    }
}

