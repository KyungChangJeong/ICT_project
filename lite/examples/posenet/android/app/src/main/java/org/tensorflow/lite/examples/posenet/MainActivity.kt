package org.tensorflow.lite.examples.posenet

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.LinearLayout

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val tutorial = findViewById<LinearLayout>(R.id.tutorial)

        tutorial.setOnClickListener({
            val intent = Intent(this, PosenetActivity::class.java)
            startActivity(intent)
        })

    }
}

