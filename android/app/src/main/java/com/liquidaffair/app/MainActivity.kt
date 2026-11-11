package com.liquidaffair.app

import android.annotation.SuppressLint
import android.app.Activity
import android.app.AlertDialog
import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.view.WindowManager
import android.webkit.HttpAuthHandler
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast

class MainActivity : Activity() {
    private lateinit var webView: WebView
    private lateinit var progress: ProgressBar
    private lateinit var loading: LinearLayout

    companion object {
        private const val APP_URL = "https://liquid-affair-app-production.up.railway.app"
        private const val BG = Color.rgb(9, 8, 18)
        private const val GOLD = Color.rgb(231, 190, 104)
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.statusBarColor = BG
        window.navigationBarColor = BG
        window.decorView.systemUiVisibility = 0
        window.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(BG)
        }
        loading = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = android.view.Gravity.CENTER
            setPadding(28, 24, 28, 24)
            setBackgroundColor(BG)
        }
        val brand = TextView(this).apply {
            text = "THE LIQUID AFFAIR"
            textSize = 20f
            setTextColor(GOLD)
            gravity = android.view.Gravity.CENTER
            letterSpacing = .12f
        }
        val subtitle = TextView(this).apply {
            text = "BAR OPERATIONS"
            textSize = 10f
            setTextColor(Color.LTGRAY)
            gravity = android.view.Gravity.CENTER
            setPadding(0, 6, 0, 20)
            letterSpacing = .18f
        }
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            isIndeterminate = true
        }
        loading.addView(brand)
        loading.addView(subtitle)
        loading.addView(progress, LinearLayout.LayoutParams(-1, 4))
        webView = WebView(this).apply {
            setBackgroundColor(BG)
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.databaseEnabled = true
            settings.loadsImagesAutomatically = true
            settings.javaScriptCanOpenWindowsAutomatically = true
            settings.setSupportMultipleWindows(false)
            settings.useWideViewPort = true
            settings.loadWithOverviewMode = false
            settings.builtInZoomControls = false
            settings.displayZoomControls = false
            settings.allowFileAccess = false
            settings.allowContentAccess = false
            webChromeClient = WebChromeClient()
            webViewClient = object : WebViewClient() {
                override fun onPageStarted(view: WebView?, url: String?, favicon: android.graphics.Bitmap?) {
                    loading.visibility = View.VISIBLE
                    progress.visibility = View.VISIBLE
                }

                override fun onPageFinished(view: WebView?, url: String?) {
                    loading.visibility = View.GONE
                    progress.visibility = View.GONE
                    // Keep the app viewport at device width; let the web UI handle responsive layout.
                    view?.evaluateJavascript(
                        "document.querySelector('meta[name=viewport]')?.setAttribute('content','width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover');",
                        null
                    )
                }

                override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                    val url = request?.url ?: return false
                    return url.scheme != "https" && url.scheme != "http"
                }

                override fun onReceivedHttpAuthRequest(view: WebView?, handler: HttpAuthHandler?, host: String?, realm: String?) {
                    val user = EditText(this@MainActivity).apply { hint = "Username" }
                    val pass = EditText(this@MainActivity).apply {
                        hint = "Password"
                        inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
                    }
                    val box = LinearLayout(this@MainActivity).apply {
                        orientation = LinearLayout.VERTICAL
                        setPadding(24, 8, 24, 0)
                        addView(user)
                        addView(pass)
                    }
                    AlertDialog.Builder(this@MainActivity)
                        .setTitle("Sign in to The Liquid Affair")
                        .setView(box)
                        .setPositiveButton("Sign in") { _, _ ->
                            handler?.proceed(user.text.toString(), pass.text.toString())
                        }
                        .setNegativeButton("Cancel") { _, _ -> handler?.cancel() }
                        .setOnCancelListener { handler?.cancel() }
                        .show()
                }
            }
        }

        root.addView(loading, LinearLayout.LayoutParams(-1, -2))
        root.addView(webView, LinearLayout.LayoutParams(-1, 0, 1f))
        setContentView(root)
        if (savedInstanceState == null) webView.loadUrl(APP_URL)
        else webView.restoreState(savedInstanceState)
    }

    override fun onSaveInstanceState(outState: Bundle) {
        webView.saveState(outState)
        super.onSaveInstanceState(outState)
    }

    @Deprecated("Deprecated in Android API")
    override fun onBackPressed() {
        if (::webView.isInitialized && webView.canGoBack()) webView.goBack()
        else super.onBackPressed()
    }

    override fun onDestroy() {
        if (::webView.isInitialized) {
            webView.stopLoading()
            webView.destroy()
        }
        super.onDestroy()
    }
}
