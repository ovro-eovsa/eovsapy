#!/bin/bash

# Define the root directory for the UDBms files
BASE_DIR="/data1/eovsa/fits/UDBms"

# Function to display help message
function display_help() {
    echo "Usage: ./archive_udbms.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dry-run       Echo the commands to be run without executing them."
    echo "  --subdir DIR    Process only the specified subdirectory (e.g., 202111)."
    echo "  --remove-ms     Remove the original .ms file after successful archiving."
    echo "  --help          Display this help message."
    echo ""
    echo "Description:"
    echo "This script archives individual UDByyyymmdd.ms files into tar.gz files."
    echo "It processes only yyyymm subdirectories under the base directory: $BASE_DIR"
    echo ""
    echo "Example:"
    echo "  ./archive_udbms.sh                    Run the script for all subdirectories."
    echo "  ./archive_udbms.sh --subdir 202111    Run the script for subdirectory 202111."
    echo "  ./archive_udbms.sh --remove-ms        Remove .ms files after archiving."
    echo "  ./archive_udbms.sh --dry-run          Dry run mode, only echo commands."
    echo ""
    exit 0
}

# Default options
DRY_RUN=false
REMOVE_MS=false
SPECIFIC_SUBDIR=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            echo "Dry run mode enabled. Commands will only be echoed, not executed."
            shift
            ;;
        --subdir)
            if [[ -n "$2" && "$2" =~ ^[0-9]{6}$ ]]; then
                SPECIFIC_SUBDIR="$2"
                echo "Processing only subdirectory: $SPECIFIC_SUBDIR"
                shift 2
            else
                echo "Error: --subdir requires a valid yyyymm argument."
                exit 1
            fi
            ;;
        --remove-ms)
            REMOVE_MS=true
            echo "Option enabled: Original .ms files will be removed after archiving."
            shift
            ;;
        --help)
            display_help
            ;;
        *)
            echo "Invalid option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Get the list of subdirectories to process
if [[ -n "$SPECIFIC_SUBDIR" ]]; then
    SUBDIRS=("$BASE_DIR/$SPECIFIC_SUBDIR/")
    if [[ ! -d "${SUBDIRS[0]}" ]]; then
        echo "Error: Specified subdirectory does not exist: $SPECIFIC_SUBDIR"
        exit 1
    fi
else
    SUBDIRS=("$BASE_DIR"/*/)
fi

# Count total number of .ms files to process
TOTAL_FILES=0
for SUBDIR in "${SUBDIRS[@]}"; do
    if [[ -d "$SUBDIR" ]]; then
        TOTAL_FILES=$((TOTAL_FILES + $(find "$SUBDIR" -maxdepth 1 -name "*.ms" | wc -l)))
    fi
done
echo "Total files to process: $TOTAL_FILES"

# Initialize progress counters
PROCESSED_FILES=0

# Process the subdirectories
for SUBDIR in "${SUBDIRS[@]}"; do
    # Extract the subdirectory name (e.g., "201701")
    DIRNAME=$(basename "$SUBDIR")

    # Check if the subdirectory name matches the yyyymm format
    if [[ "$DIRNAME" =~ ^[0-9]{6}$ ]]; then
        echo "Processing directory: $SUBDIR"

        # Loop through each UDByyyymmdd.ms file in the subdirectory
        for MS_FILE in "$SUBDIR"/*.ms; do
            PROCESSED_FILES=$((PROCESSED_FILES + 1))
            # Check if there are any .ms files in the current directory
            if [ -e "$MS_FILE" ]; then
                # Extract the base name of the file (e.g., UDB20211130.ms)
                BASENAME=$(basename "$MS_FILE")
                # Generate the tar.gz name, keeping the same base name
                TAR_NAME="${BASENAME}.tar.gz"
                TAR_PATH="$SUBDIR$TAR_NAME"

                # Check if the tar.gz file already exists
                if [ -e "$TAR_PATH" ]; then
                    if $DRY_RUN; then
                        echo "[Dry Run] Progress [$PROCESSED_FILES/$TOTAL_FILES]: Tar file already exists, skipping: $TAR_PATH"
                    else
                        echo "Progress [$PROCESSED_FILES/$TOTAL_FILES]: Tar file already exists, skipping: $TAR_PATH"
                    fi
                else
                    # Create the tar.gz archive command, ensuring only the basename is included
                    if $DRY_RUN; then
                        echo "[Dry Run] Progress [$PROCESSED_FILES/$TOTAL_FILES]: tar -cf - -C \"$SUBDIR\" \"$BASENAME\" | pv | gzip > \"$TAR_PATH\""
                    else
                        # Execute the tar command with progress using pv
                        echo "Progress [$PROCESSED_FILES/$TOTAL_FILES]: Archiving $BASENAME to $TAR_PATH"
                        tar -cf - -C "$SUBDIR" "$BASENAME" | pv -pterb -s $(du -sb "$MS_FILE" | awk '{print $1}') | gzip > "$TAR_PATH"
                    fi
                fi
                # Remove the .ms file if the archive was created successfully
                if $REMOVE_MS; then
                  if [ -e "$MS_FILE" ]; then
                    if $DRY_RUN; then
                      echo "[Dry Run] rm -rf $MS_FILE"
                    else
                      echo "Removing original .ms file: $MS_FILE"
                      rm -rf "$MS_FILE"
                    fi
                  else
                    if $DRY_RUN; then
                      echo "[Dry Run] Original .ms file not found, skipping: $MS_FILE"
                    else
                      echo "Original .ms file not found, skipping: $MS_FILE"
                    fi
                  fi
                fi
            fi
        done
    else
        echo "Skipping non-yyyymm directory: $SUBDIR"
    fi
done

echo -e "\nArchiving completed. Processed $PROCESSED_FILES out of $TOTAL_FILES files."
