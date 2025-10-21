#!/bin/bash
# FUREcast - Automated Update Scheduler Setup
#
# This script helps set up automated scheduling for data updates and model retraining.
# It provides options for cron-based scheduling on Linux systems.

# Cron schedule examples:
# - Every 3 hours during market hours (9am-4pm ET): 0 9,12,15 * * 1-5
# - Daily at 6pm ET: 0 18 * * 1-5
# - Every weekday at 8pm ET: 0 20 * * 1-5

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
UPDATE_SCRIPT="$SCRIPT_DIR/update_and_retrain.py"
PYTHON_PATH=$(which python)

echo "============================================================"
echo "  FUREcast - Update Scheduler Setup"
echo "============================================================"
echo ""

# Check if update script exists
if [ ! -f "$UPDATE_SCRIPT" ]; then
    echo "✗ Error: Update script not found at $UPDATE_SCRIPT"
    exit 1
fi

echo "✓ Update script found: $UPDATE_SCRIPT"
echo "✓ Python path: $PYTHON_PATH"
echo ""

# Function to add cron job
add_cron_job() {
    local schedule="$1"
    local description="$2"
    local quick_flag="$3"
    
    # Build command
    local command="cd $SCRIPT_DIR && $PYTHON_PATH $UPDATE_SCRIPT $quick_flag >> $SCRIPT_DIR/logs/cron_update.log 2>&1"
    local cron_line="$schedule $command"
    
    echo ""
    echo "Cron job to add:"
    echo "  Schedule: $description"
    echo "  Command: $command"
    echo ""
    
    # Check if already exists
    if crontab -l 2>/dev/null | grep -q "$UPDATE_SCRIPT"; then
        echo "⚠️  A cron job for this script already exists"
        echo ""
        crontab -l | grep "$UPDATE_SCRIPT"
        echo ""
        read -p "Replace existing cron job? (y/n): " replace
        
        if [ "$replace" != "y" ]; then
            echo "Cancelled"
            return
        fi
        
        # Remove old entries
        (crontab -l 2>/dev/null | grep -v "$UPDATE_SCRIPT") | crontab -
    fi
    
    # Add new cron job
    (crontab -l 2>/dev/null; echo "$cron_line") | crontab -
    
    echo "✓ Cron job added successfully"
}

# Main menu
echo "Choose a scheduling option:"
echo ""
echo "  1) Daily at 6:00 PM ET (recommended for daily retraining)"
echo "  2) Every 3 hours during market hours (9am, 12pm, 3pm, 6pm ET)"
echo "  3) Twice daily (8am and 6pm ET)"
echo "  4) Custom schedule"
echo "  5) View current cron jobs"
echo "  6) Remove existing cron job"
echo "  7) Manual test (run now)"
echo "  8) Exit"
echo ""

read -p "Select option (1-8): " choice

case $choice in
    1)
        add_cron_job "0 18 * * 1-5" "Daily at 6:00 PM ET (Mon-Fri)" ""
        ;;
    2)
        add_cron_job "0 9,12,15,18 * * 1-5" "Every 3 hours during market hours" "--quick"
        ;;
    3)
        add_cron_job "0 8,18 * * 1-5" "Twice daily at 8am and 6pm ET" ""
        ;;
    4)
        echo ""
        echo "Enter cron schedule (e.g., '0 18 * * 1-5' for daily at 6pm Mon-Fri):"
        read -p "Schedule: " custom_schedule
        read -p "Use quick training? (y/n): " use_quick
        
        quick_flag=""
        if [ "$use_quick" = "y" ]; then
            quick_flag="--quick"
        fi
        
        add_cron_job "$custom_schedule" "Custom: $custom_schedule" "$quick_flag"
        ;;
    5)
        echo ""
        echo "Current cron jobs:"
        echo ""
        if crontab -l 2>/dev/null | grep -q "$UPDATE_SCRIPT"; then
            crontab -l | grep "$UPDATE_SCRIPT"
        else
            echo "  (No cron jobs found for this script)"
        fi
        echo ""
        ;;
    6)
        echo ""
        if crontab -l 2>/dev/null | grep -q "$UPDATE_SCRIPT"; then
            echo "Current cron job(s):"
            crontab -l | grep "$UPDATE_SCRIPT"
            echo ""
            read -p "Remove this cron job? (y/n): " confirm
            
            if [ "$confirm" = "y" ]; then
                (crontab -l 2>/dev/null | grep -v "$UPDATE_SCRIPT") | crontab -
                echo "✓ Cron job removed"
            else
                echo "Cancelled"
            fi
        else
            echo "No cron jobs found for this script"
        fi
        echo ""
        ;;
    7)
        echo ""
        echo "Running manual test..."
        read -p "Use quick training? (y/n): " use_quick
        
        quick_flag=""
        if [ "$use_quick" = "y" ]; then
            quick_flag="--quick"
        fi
        
        echo ""
        cd "$SCRIPT_DIR" || exit 1
        $PYTHON_PATH "$UPDATE_SCRIPT" $quick_flag
        ;;
    8)
        echo "Exiting"
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  Setup Complete"
echo "============================================================"
echo ""
echo "Useful commands:"
echo "  • View cron jobs: crontab -l"
echo "  • Edit cron jobs: crontab -e"
echo "  • View cron logs: tail -f $SCRIPT_DIR/logs/cron_update.log"
echo "  • Run manually: python $UPDATE_SCRIPT [--quick] [--force]"
echo ""
