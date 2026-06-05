#!/bin/bash
set -euo pipefail

# ============================================================
# CONFIGURATION - EDIT THESE
# ============================================================

APP_NAME="Tractome"
VOLNAME="Tractome"

# Path to the .app produced by Briefcase.
# IMPORTANT: This can be inside Google Drive, but the script will copy it locally before signing.
APP_SRC="/Users/paolo/Library/CloudStorage/GoogleDrive-avesani@fbk.eu/My Drive/Projects/NeuVirCarv_2025/Tractome2/code/build/tractome/macos/app/Tractome.app"

# Your valid Developer ID Application identity.
IDENTITY="Developer ID Application: Fondazione Bruno Kessler (X3HGX9BUQ8)"

# Output folder.
WORK="$HOME/Downloads/tractome-release"

# Set to 1 if your PySide6/Python app needs library validation disabled.
# For PySide6 apps, this is often needed.
USE_ENTITLEMENTS=1

# ============================================================
# INTERNAL PATHS
# ============================================================

APP="$WORK/$APP_NAME.app"
DMGROOT="$WORK/dmgroot"
RW_DMG="$WORK/$APP_NAME-rw.dmg"
FINAL_DMG="$WORK/$APP_NAME.dmg"
ENTITLEMENTS="$WORK/entitlements.plist"

# ============================================================
# CHECKS
# ============================================================

echo "Checking signing identity..."
security find-identity -v -p codesigning | grep "$IDENTITY" >/dev/null || {
    echo "ERROR: Signing identity not found:"
    echo "$IDENTITY"
    echo
    echo "Available identities:"
    security find-identity -v -p codesigning
    exit 1
}

if [ ! -d "$APP_SRC" ]; then
    echo "ERROR: APP_SRC does not exist:"
    echo "$APP_SRC"
    exit 1
fi

# ============================================================
# PREPARE LOCAL COPY
# ============================================================

echo "Cleaning output folder..."
rm -rf "$WORK"
mkdir -p "$WORK"

#echo "Copying app to local signing folder..."
#ditto "$APP_SRC" "$APP"

echo "Copying app to local signing folder with rsync..."
rm -rf "$APP"
rsync -a --delete "$APP_SRC/" "$APP/"

echo "Removing extended attributes..."
xattr -cr "$APP"

echo "Removing Python object files not needed at runtime..."
find "$APP" -type f -name "*.o" -print -delete

# ============================================================
# ENTITLEMENTS
# ============================================================

if [ "$USE_ENTITLEMENTS" = "1" ]; then
    echo "Creating entitlements file..."
    cat > "$ENTITLEMENTS" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "https://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>

    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>

    <key>com.apple.security.cs.allow-jit</key>
    <true/>

    <key>com.apple.security.cs.allow-dyld-environment-variables</key>
    <true/>
</dict>
</plist>
EOF
fi

# ============================================================
# SIGN NESTED CODE
# ============================================================

echo "Signing .dylib and .so files..."
find "$APP" -type f \( -name "*.dylib" -o -name "*.so" \) -print0 | while IFS= read -r -d '' f; do
    echo "Signing: $f"
    codesign --force --options runtime --timestamp --sign "$IDENTITY" "$f"
done

echo "Signing .framework bundles..."
find "$APP" -type d -name "*.framework" -print0 | while IFS= read -r -d '' fw; do
    echo "Signing framework: $fw"
    codesign --force --options runtime --timestamp --sign "$IDENTITY" "$fw"
done

echo "Signing other executable Mach-O files outside frameworks..."
find "$APP" -type f -perm -111 -print0 | while IFS= read -r -d '' f; do
    case "$f" in
        *.framework/*)
            continue
            ;;
    esac

    if file "$f" | grep -q "Mach-O"; then
        echo "Signing executable: $f"
        codesign --force --options runtime --timestamp --sign "$IDENTITY" "$f"
    fi
done

# ============================================================
# SIGN MAIN APP
# ============================================================

echo "Signing main app bundle..."

if [ "$USE_ENTITLEMENTS" = "1" ]; then
    codesign --force \
        --options runtime \
        --timestamp \
        --entitlements "$ENTITLEMENTS" \
        --sign "$IDENTITY" \
        "$APP"
else
    codesign --force \
        --options runtime \
        --timestamp \
        --sign "$IDENTITY" \
        "$APP"
fi

echo "Verifying signed app..."
codesign --verify --deep --strict --verbose=4 "$APP"

echo
echo "Testing Gatekeeper assessment for app..."
spctl -a -vvv -t exec "$APP" || true

# ============================================================
# CREATE BRIEFCASE-STYLE DMG ROOT
# ============================================================

echo "Creating DMG root..."
rm -rf "$DMGROOT"
mkdir -p "$DMGROOT"

ditto "$APP" "$DMGROOT/$APP_NAME.app"
ln -s /Applications "$DMGROOT/Applications"

# ============================================================
# CREATE READ-WRITE DMG
# ============================================================

echo "Creating read-write DMG..."
rm -f "$RW_DMG" "$FINAL_DMG"

hdiutil create \
    -volname "$VOLNAME" \
    -srcfolder "$DMGROOT" \
    -ov \
    -format UDRW \
    "$RW_DMG"

# ============================================================
# MOUNT DMG AND SET FINDER LAYOUT
# ============================================================

echo "Mounting read-write DMG..."
MOUNT_DIR=$(mktemp -d /tmp/tractome-dmg.XXXXXX)

hdiutil attach "$RW_DMG" \
    -mountpoint "$MOUNT_DIR" \
    -readwrite \
    -noverify \
    -noautoopen

sleep 2

echo "Setting Finder window layout..."
osascript <<EOF
set dmgPath to POSIX file "$MOUNT_DIR" as alias

tell application "Finder"
    open dmgPath
    delay 2

    set dmgWindow to container window of dmgPath

    set current view of dmgWindow to icon view
    set toolbar visible of dmgWindow to false
    set statusbar visible of dmgWindow to false
    set bounds of dmgWindow to {100, 100, 650, 420}

    set viewOptions to the icon view options of dmgWindow
    set arrangement of viewOptions to not arranged
    set icon size of viewOptions to 96

    set position of item "$APP_NAME.app" of dmgPath to {170, 160}
    set position of item "Applications" of dmgPath to {430, 160}

    update dmgPath without registering applications
    delay 2
    close dmgWindow
end tell
EOF

sync

echo "Detaching DMG..."
hdiutil detach "$MOUNT_DIR"

# ============================================================
# CONVERT TO COMPRESSED READ-ONLY DMG
# ============================================================

echo "Converting to compressed read-only DMG..."
hdiutil convert "$RW_DMG" \
    -format UDZO \
    -imagekey zlib-level=9 \
    -o "$FINAL_DMG"

rm -f "$RW_DMG"

# ============================================================
# SIGN FINAL DMG
# ============================================================

echo "Signing final DMG..."
codesign --force \
    --timestamp \
    --sign "$IDENTITY" \
    "$FINAL_DMG"

echo "Verifying DMG signature..."
codesign --verify --verbose=4 "$FINAL_DMG"

echo
echo "Gatekeeper assessment for DMG:"
spctl -a -t open --context context:primary-signature -v "$FINAL_DMG" || true

# ============================================================
# DONE
# ============================================================

echo
echo "Done."
echo "Signed app:"
echo "$APP"
echo
echo "Signed DMG:"
echo "$FINAL_DMG"
echo
echo "Next step, if you want public distribution:"
echo "xcrun notarytool submit \"$FINAL_DMG\" --keychain-profile \"notary-profile\" --wait"
echo "xcrun stapler staple \"$FINAL_DMG\""
echo "xcrun stapler validate \"$FINAL_DMG\""
