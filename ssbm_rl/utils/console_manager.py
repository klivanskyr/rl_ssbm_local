"""
Console manager for creating and managing Dolphin instances
"""

import melee
import signal
import sys


class ConsoleManager:
    """
    Manages Dolphin console lifecycle
    
    Handles:
    - Console creation with proper configuration
    - Controller setup
    - Signal handling for clean shutdown
    """
    
    def __init__(self,
                 dolphin_path="/opt/slippi-extracted/AppRun",
                 iso_path="/opt/melee/Melee.iso",
                 save_replays=False,
                 enable_logging=False):
        """
        Args:
            dolphin_path: Path to Dolphin executable
            iso_path: Path to Melee ISO
            save_replays: Whether to save .slp replays
            enable_logging: Whether to enable libmelee logging
        """
        self.dolphin_path = dolphin_path
        self.iso_path = iso_path
        self.save_replays = save_replays
        self.enable_logging = enable_logging
        
        self.console = None
        self.controllers = {}
        self.logger = None
    
    def create_console(self):
        """Create and configure console"""
        # Create logger if enabled
        if self.enable_logging:
            self.logger = melee.Logger()
        
        # Create console
        self.console = melee.Console(
            path=self.dolphin_path,
            slippi_address='127.0.0.1',
            logger=self.logger,
            fullscreen=False,
            gfx_backend='Null',
            disable_audio=True,
            save_replays=self.save_replays
        )
        
        print(f"Console created at {self.dolphin_path}")
        return self.console
    
    def add_controller(self, port, controller_type=melee.ControllerType.STANDARD):
        """
        Add controller at specified port
        
        Args:
            port: Controller port number (1-4)
            controller_type: Type of controller
            
        Returns:
            Controller instance
        """
        if self.console is None:
            raise RuntimeError("Console not created. Call create_console() first.")
        
        controller = melee.Controller(
            console=self.console,
            port=port,
            type=controller_type
        )
        
        self.controllers[port] = controller
        print(f"Controller added at port {port}")
        return controller
    
    def start(self):
        """Start console and connect controllers"""
        if self.console is None:
            raise RuntimeError("Console not created. Call create_console() first.")
        
        # Run console
        print(f"Starting Dolphin with ISO: {self.iso_path}")
        self.console.run(iso_path=self.iso_path)
        
        # Connect to console
        print("Connecting to console...")
        if not self.console.connect():
            raise RuntimeError("Failed to connect to console")
        print("Console connected")
        
        # Connect all controllers
        for port, controller in self.controllers.items():
            print(f"Connecting controller at port {port}...")
            if not controller.connect():
                raise RuntimeError(f"Failed to connect controller at port {port}")
            print(f"Controller {port} connected")
        
        return True
    
    def stop(self):
        """Stop console and cleanup"""
        if self.logger:
            self.logger.writelog()
            print(f"Log written to: {self.logger.filename}")
        
        if self.console:
            self.console.stop()
            print("Console stopped")
    
    def setup_signal_handler(self):
        """Setup signal handler for clean shutdown"""
        def signal_handler(sig, frame):
            print("\n\nReceived shutdown signal...")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
