# Android telemetry client

This small Android application is a **research telemetry simulator**, not a real UPI application. It can:

1. send a payment-journey telemetry event to `POST /telemetry`;
2. call `POST /precheck` when a recipient is selected;
3. call `POST /score` immediately before a simulated payment;
4. simulate first-time-payee, QR, PIN-reset and remote-access risk flags.

## Run

1. Open the `android-client` directory in Android Studio Quail 2/3.
2. Let Gradle sync.
3. Start the Python API on your computer at port 8000.
4. On the Android emulator keep the API URL as `http://10.0.2.2:8000`.
5. On a physical phone replace it with your computer's LAN address, for example `http://192.168.1.10:8000`.

The app uses coarse/current location only when permission is granted. `remote_access_indicator` is deliberately a simulation toggle because an ordinary Android application cannot reliably inspect every other installed/running app. A real Play Integrity integration requires a Google Play Console project and server-side token verification; the prototype sends a boolean placeholder.
