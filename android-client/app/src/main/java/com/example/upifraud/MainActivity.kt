package com.example.upifraud

import android.Manifest
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationManager
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.upifraud.databinding.ActivityMainBinding
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONObject
import java.io.IOException
import java.time.Instant
import java.util.UUID

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val client = OkHttpClient()
    private var currentLocation: Location? = null

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { refreshLocation() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        requestLocationPermission()

        binding.buttonTelemetry.setOnClickListener {
            sendJson("/telemetry", telemetryPayload("RECIPIENT_NAME_VIEWED"))
        }
        binding.buttonPrecheck.setOnClickListener {
            sendJson("/precheck", transactionPayload())
        }
        binding.buttonScore.setOnClickListener {
            sendJson("/score", transactionPayload())
        }
    }

    private fun requestLocationPermission() {
        val fine = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
        if (fine != PackageManager.PERMISSION_GRANTED) {
            permissionLauncher.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION))
        } else {
            refreshLocation()
        }
    }

    private fun refreshLocation() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED
        ) return
        val manager = getSystemService(LOCATION_SERVICE) as LocationManager
        currentLocation = listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER)
            .mapNotNull { provider -> runCatching { manager.getLastKnownLocation(provider) }.getOrNull() }
            .maxByOrNull { it.time }
    }

    private fun telemetryPayload(eventType: String): JSONObject = JSONObject().apply {
        put("event_id", UUID.randomUUID().toString())
        put("user_id", binding.editPayer.text.toString())
        put("session_id", "android-demo-session")
        put("timestamp", Instant.now().toString())
        put("event_type", eventType)
        put("device_id", "android-demo-device")
        currentLocation?.let {
            put("latitude", it.latitude)
            put("longitude", it.longitude)
        }
        put("amount", binding.editAmount.text.toString().toDoubleOrNull() ?: 0.0)
        put("recipient_id", binding.editPayee.text.toString())
        put("metadata", JSONObject().put("review_seconds", 5))
    }

    private fun transactionPayload(): JSONObject = JSONObject().apply {
        put("transaction_id", "ANDROID-${System.currentTimeMillis()}")
        put("timestamp", Instant.now().toString())
        put("payer_id", binding.editPayer.text.toString())
        put("payee_id", binding.editPayee.text.toString())
        put("amount", binding.editAmount.text.toString().toDoubleOrNull() ?: 1.0)
        put("payment_mode", "QR")
        put("device_id", "android-demo-device")
        put("vpa_id", "demo@upi")
        put("qr_id", "android-demo-qr")
        currentLocation?.let {
            put("latitude", it.latitude)
            put("longitude", it.longitude)
            put("payee_latitude", it.latitude)
            put("payee_longitude", it.longitude)
        }
        put("is_new_payee", binding.checkNewPayee.isChecked)
        put("qr_verified", binding.checkQrVerified.isChecked)
        put("collect_request", false)
        put("merchant_category", "TRANSPORT")
        put("device_age_days", 120)
        put("sim_age_days", 500)
        put("app_registration_age_days", 90)
        put("device_known", true)
        put("play_integrity_ok", true)
        put("recent_pin_reset", binding.checkPinReset.isChecked)
        put("app_reregistered", false)
        put("remote_access_indicator", binding.checkRemoteAccess.isChecked)
        put("overlay_indicator", false)
        put("recipient_complaint_score", 0.0)
        put("known_mule", false)
        put("common_owner_verified", false)
    }

    private fun sendJson(path: String, payload: JSONObject) {
        val baseUrl = binding.editApiUrl.text.toString().trimEnd('/')
        val request = Request.Builder()
            .url(baseUrl + path)
            .post(payload.toString().toRequestBody("application/json".toMediaType()))
            .build()
        binding.textResult.text = "Sending request..."
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread { binding.textResult.text = "Request failed: ${e.message}" }
            }

            override fun onResponse(call: Call, response: Response) {
                val text = response.body?.string() ?: "No body"
                runOnUiThread { binding.textResult.text = "HTTP ${response.code}\n$text" }
            }
        })
    }
}
