import cv2
import mediapipe as mp
import numpy as np
import base64
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GestureType(Enum):
    """Enumeration of different gesture types"""
    UNKNOWN = "unknown"
    OPEN_PALM = "open_palm"
    CLOSED_FIST = "closed_fist"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    POINTING = "pointing"
    OK_SIGN = "ok_sign"
    HELP_SIGNAL = "help_signal"
    SOS = "sos"
    STOP = "stop"
    CALL_ME = "call_me"
    I_LOVE_YOU = "i_love_you"
    ROCK_ON = "rock_on"
    GUN_GESTURE = "gun_gesture"
    KNIFE_THREAT = "knife_threat"
    CHOKING_SIGNAL = "choking_signal"
    THUMBS_DOWN = "thumbs_down"
    MIDDLE_FINGER = "middle_finger"
    HANG_LOOSE = "hang_loose"

@dataclass
class HandLandmarks:
    """Data class to store hand landmark information"""
    landmarks: List[Tuple[float, float]]
    handedness: str  # 'Left' or 'Right'
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height

@dataclass
class GestureResult:
    """Data class to store gesture recognition results"""
    gesture_type: GestureType
    confidence: float
    hand_landmarks: Optional[HandLandmarks]
    is_emergency: bool
    description: str
    visual_frame: Optional[np.ndarray] = None

class WorkingGestureRecognizer:
    """A complete working gesture recognition system using MediaPipe"""
    
    def __init__(self):
        """Initialize the gesture recognizer with MediaPipe"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Hands with improved settings
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,  # Lower threshold for better detection
            min_tracking_confidence=0.5
        )
        
        # Hand landmark connections for drawing
        self.hand_connections = self.mp_hands.HAND_CONNECTIONS
        
        logger.info("WorkingGestureRecognizer initialized successfully")
    
    def decode_base64_image(self, base64_string: str) -> Optional[np.ndarray]:
        """Decode base64 image string to OpenCV format"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Failed to decode image")
                return None
                
            return image
            
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            return None
    
    def extract_hand_landmarks(self, image: np.ndarray) -> List[HandLandmarks]:
        """Extract hand landmarks from image using MediaPipe"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.hands.process(rgb_image)
            
            hand_landmarks_list = []
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Extract landmark coordinates
                    landmarks = []
                    h, w, _ = image.shape
                    
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        landmarks.append((x, y))
                    
                    # Calculate bounding box
                    x_coords = [lm[0] for lm in landmarks]
                    y_coords = [lm[1] for lm in landmarks]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Add padding
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    
                    bounding_box = (x_min, y_min, x_max - x_min, y_max - y_min)
                    
                    # Get handedness
                    hand_label = handedness.classification[0].label
                    confidence = handedness.classification[0].score
                    
                    hand_landmarks_obj = HandLandmarks(
                        landmarks=landmarks,
                        handedness=hand_label,
                        confidence=confidence,
                        bounding_box=bounding_box
                    )
                    
                    hand_landmarks_list.append(hand_landmarks_obj)
            
            return hand_landmarks_list
            
        except Exception as e:
            logger.error(f"Error extracting hand landmarks: {e}")
            return []
    
    def draw_landmarks_and_connections(self, image: np.ndarray, hand_landmarks_list: List[HandLandmarks]) -> np.ndarray:
        """Draw hand landmarks and connections on the image"""
        try:
            annotated_image = image.copy()
            
            for hand_landmarks_obj in hand_landmarks_list:
                landmarks = hand_landmarks_obj.landmarks
                
                # Draw connections
                for connection in self.hand_connections:
                    start_idx, end_idx = connection
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        start_point = landmarks[start_idx]
                        end_point = landmarks[end_idx]
                        
                        cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)
                
                # Draw landmarks
                for i, (x, y) in enumerate(landmarks):
                    # Different colors for different landmark types
                    if i in [4, 8, 12, 16, 20]:  # Fingertips
                        color = (0, 0, 255)  # Red
                        radius = 6
                    elif i == 0:  # Wrist
                        color = (255, 0, 0)  # Blue
                        radius = 8
                    else:
                        color = (255, 255, 0)  # Cyan
                        radius = 4
                    
                    cv2.circle(annotated_image, (x, y), radius, color, -1)
                    cv2.circle(annotated_image, (x, y), radius + 1, (0, 0, 0), 1)
                
                # Draw bounding box
                x, y, w, h = hand_landmarks_obj.bounding_box
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 255), 2)
                
                # Add handedness label
                label = f"{hand_landmarks_obj.handedness} ({hand_landmarks_obj.confidence:.2f})"
                cv2.putText(annotated_image, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return annotated_image
            
        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")
            return image
    
    def analyze_finger_states(self, landmarks: List[Tuple[float, float]]) -> Dict[str, bool]:
        """Analyze which fingers are extended"""
        try:
            finger_states = {
                'thumb': False,
                'index': False,
                'middle': False,
                'ring': False,
                'pinky': False
            }
            
            if len(landmarks) < 21:
                return finger_states
            
            # Thumb detection - compare tip (4) with MCP joint (2)
            # For right hand: thumb extended if tip is to the right of MCP
            # For left hand: thumb extended if tip is to the left of MCP
            thumb_tip_x = landmarks[4][0]
            thumb_mcp_x = landmarks[2][0]
            wrist_x = landmarks[0][0]
            
            # Determine hand chirality based on thumb position relative to wrist
            if thumb_tip_x > wrist_x:  # Right hand
                finger_states['thumb'] = thumb_tip_x > thumb_mcp_x
            else:  # Left hand
                finger_states['thumb'] = thumb_tip_x < thumb_mcp_x
            
            # Other fingers (compare tip with previous joint)
            finger_tips = [8, 12, 16, 20]
            finger_pips = [6, 10, 14, 18]
            finger_names = ['index', 'middle', 'ring', 'pinky']
            
            for tip, pip, name in zip(finger_tips, finger_pips, finger_names):
                if landmarks[tip][1] < landmarks[pip][1]:  # Tip is above PIP joint
                    finger_states[name] = True
            
            return finger_states
            
        except Exception as e:
            logger.error(f"Error analyzing finger states: {e}")
            return {'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': False}
    
    def _is_choking_gesture(self, landmarks: List[Tuple[float, float]]) -> bool:
        """Detect if hands are positioned around neck area (choking signal)"""
        try:
            if len(landmarks) < 21:
                return False
            
            # Check if hand is positioned in upper chest/neck area
            # This is a simplified check - in practice, you'd need both hands
            wrist_y = landmarks[0][1]
            middle_finger_tip_y = landmarks[12][1]
            
            # If hand is positioned high (neck/chest area) and fingers are curved
            # This is a basic approximation
            return wrist_y < 200 and middle_finger_tip_y < wrist_y
            
        except Exception as e:
            logger.error(f"Error detecting choking gesture: {e}")
            return False
    
    def _is_knife_threat(self, finger_states: Dict[str, bool], landmarks: List[Tuple[float, float]]) -> bool:
        """Detect knife threat gesture (closed fist with aggressive positioning)"""
        try:
            if len(landmarks) < 21:
                return False
            
            # Check for closed fist with specific hand orientation
            extended_fingers = sum(finger_states.values())
            
            # Knife threat: mostly closed fist (0-1 fingers) with aggressive angle
            if extended_fingers <= 1:
                # Check hand angle/orientation for threatening posture
                wrist = landmarks[0]
                middle_finger_mcp = landmarks[9]
                
                # Simple angle check - more sophisticated detection would be needed
                angle_indicator = abs(wrist[1] - middle_finger_mcp[1])
                return angle_indicator > 20  # Arbitrary threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting knife threat: {e}")
            return False
    
    def classify_gesture(self, hand_landmarks_obj: HandLandmarks) -> GestureType:
        """Classify the gesture based on hand landmarks"""
        try:
            landmarks = hand_landmarks_obj.landmarks
            finger_states = self.analyze_finger_states(landmarks)
            extended_fingers = sum(finger_states.values())
            
            # Debug logging for gesture classification
            logger.debug(f"Finger states: {finger_states}, Extended fingers: {extended_fingers}")
            
            # Enhanced gesture classification logic
            if extended_fingers == 5:
                # Check if it's a help signal (open palm facing camera)
                # For now, treat all open palms as potential help signals
                return GestureType.HELP_SIGNAL
            elif extended_fingers == 0:
                return GestureType.CLOSED_FIST
            elif extended_fingers == 1:
                if finger_states['thumb']:
                    # Check thumb direction for thumbs up vs thumbs down
                    # This is a simplified check - would need more sophisticated analysis
                    return GestureType.THUMBS_UP
                elif finger_states['index']:
                    return GestureType.POINTING
                elif finger_states['middle']:
                    return GestureType.MIDDLE_FINGER
                else:
                    return GestureType.UNKNOWN
            elif extended_fingers == 2:
                if finger_states['index'] and finger_states['middle']:
                    return GestureType.PEACE
                elif finger_states['thumb'] and finger_states['pinky']:
                    # Check if it's Call Me or Hang Loose (same finger pattern)
                    return GestureType.CALL_ME  # or HANG_LOOSE - same gesture pattern
                elif finger_states['thumb'] and finger_states['index']:
                    # Gun gesture - thumb and index finger extended (like pointing a gun)
                    return GestureType.GUN_GESTURE
                else:
                    return GestureType.UNKNOWN
            elif extended_fingers == 3:
                if finger_states['index'] and finger_states['middle'] and finger_states['ring']:
                    # Three fingers could be SOS signal
                    return GestureType.SOS
                elif finger_states['thumb'] and finger_states['index'] and finger_states['pinky']:
                    # I Love You sign (ASL)
                    return GestureType.I_LOVE_YOU
                elif finger_states['index'] and finger_states['middle'] and finger_states['pinky']:
                    # Rock on / Devil horns
                    return GestureType.ROCK_ON
                else:
                    return GestureType.UNKNOWN
            elif extended_fingers == 4:
                if not finger_states['thumb']:
                    # Four fingers could be stop signal
                    return GestureType.STOP
                else:
                    return GestureType.UNKNOWN
            else:
                # Check for special gestures that don't follow simple finger counting
                # Choking signal - hands around neck area (detected by hand position)
                if self._is_choking_gesture(landmarks):
                    return GestureType.CHOKING_SIGNAL
                # Knife threat - closed fist with aggressive positioning
                elif self._is_knife_threat(finger_states, landmarks):
                    return GestureType.KNIFE_THREAT
                else:
                    return GestureType.UNKNOWN
                
        except Exception as e:
            logger.error(f"Error classifying gesture: {e}")
            return GestureType.UNKNOWN
    
    def is_emergency_gesture(self, gesture_type: GestureType) -> bool:
        """Determine if the gesture is an emergency signal"""
        emergency_gestures = {
            GestureType.HELP_SIGNAL,
            GestureType.SOS,
            GestureType.STOP,
            GestureType.GUN_GESTURE,
            GestureType.KNIFE_THREAT,
            GestureType.CHOKING_SIGNAL
        }
        return gesture_type in emergency_gestures
    
    def get_gesture_description(self, gesture_type: GestureType) -> str:
        """Get description for the gesture"""
        descriptions = {
            GestureType.UNKNOWN: "Unknown gesture detected",
            GestureType.OPEN_PALM: "Open palm - possible greeting or help signal",
            GestureType.CLOSED_FIST: "Closed fist detected",
            GestureType.PEACE: "Peace sign detected",
            GestureType.THUMBS_UP: "Thumbs up - positive gesture",
            GestureType.THUMBS_DOWN: "Thumbs down - negative gesture",
            GestureType.POINTING: "Pointing gesture detected",
            GestureType.OK_SIGN: "OK sign detected",
            GestureType.HELP_SIGNAL: "HELP SIGNAL DETECTED - Emergency assistance needed",
            GestureType.SOS: "SOS SIGNAL DETECTED - Emergency distress call",
            GestureType.STOP: "STOP SIGNAL DETECTED - Immediate attention required",
            GestureType.CALL_ME: "Call me gesture - thumb and pinky extended",
            GestureType.I_LOVE_YOU: "I Love You sign (ASL) - thumb, index, and pinky extended",
            GestureType.ROCK_ON: "Rock on gesture - index, middle, and pinky extended",
            GestureType.GUN_GESTURE: "GUN GESTURE DETECTED - POTENTIAL THREAT - Immediate security alert",
            GestureType.KNIFE_THREAT: "KNIFE THREAT DETECTED - POTENTIAL WEAPON THREAT - Emergency response needed",
            GestureType.CHOKING_SIGNAL: "CHOKING SIGNAL DETECTED - Medical emergency - Immediate assistance required",
            GestureType.MIDDLE_FINGER: "Offensive gesture detected",
            GestureType.HANG_LOOSE: "Hang loose gesture - thumb and pinky extended (relaxed)"
        }
        return descriptions.get(gesture_type, "Unknown gesture")
    
    def recognize_gesture(self, image_input) -> GestureResult:
        """Main method to recognize gestures from image input"""
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # Base64 string
                image = self.decode_base64_image(image_input)
                if image is None:
                    return GestureResult(
                        gesture_type=GestureType.UNKNOWN,
                        confidence=0.0,
                        hand_landmarks=None,
                        is_emergency=False,
                        description="Failed to decode image"
                    )
            elif isinstance(image_input, np.ndarray):
                # NumPy array
                image = image_input
            else:
                return GestureResult(
                    gesture_type=GestureType.UNKNOWN,
                    confidence=0.0,
                    hand_landmarks=None,
                    is_emergency=False,
                    description="Invalid image input type"
                )
            
            # Extract hand landmarks
            hand_landmarks_list = self.extract_hand_landmarks(image)
            
            if not hand_landmarks_list:
                return GestureResult(
                    gesture_type=GestureType.UNKNOWN,
                    confidence=0.0,
                    hand_landmarks=None,
                    is_emergency=False,
                    description="No hands detected in image"
                )
            
            # Use the first detected hand for gesture classification
            primary_hand = hand_landmarks_list[0]
            
            # Classify gesture
            gesture_type = self.classify_gesture(primary_hand)
            
            # Check if emergency
            is_emergency = self.is_emergency_gesture(gesture_type)
            
            # Get description
            description = self.get_gesture_description(gesture_type)
            
            # Draw landmarks and connections
            visual_frame = self.draw_landmarks_and_connections(image, hand_landmarks_list)
            
            # Add gesture label to image
            if visual_frame is not None:
                label_text = f"Gesture: {gesture_type.value.upper()}"
                if is_emergency:
                    label_text = f"EMERGENCY: {gesture_type.value.upper()}"
                    color = (0, 0, 255)  # Red for emergency
                else:
                    color = (0, 255, 0)  # Green for normal
                
                cv2.putText(visual_frame, label_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Add confidence score
                conf_text = f"Confidence: {primary_hand.confidence:.2f}"
                cv2.putText(visual_frame, conf_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return GestureResult(
                gesture_type=gesture_type,
                confidence=primary_hand.confidence,
                hand_landmarks=primary_hand,
                is_emergency=is_emergency,
                description=description,
                visual_frame=visual_frame
            )
            
        except Exception as e:
            logger.error(f"Error in gesture recognition: {e}")
            return GestureResult(
                gesture_type=GestureType.UNKNOWN,
                confidence=0.0,
                hand_landmarks=None,
                is_emergency=False,
                description=f"Error during recognition: {str(e)}"
            )
    
    def detect_emergency_gestures(self, image_input) -> Dict[str, Any]:
        """Detect emergency gestures specifically"""
        result = self.recognize_gesture(image_input)
        
        return {
            'emergency_detected': result.is_emergency,
            'gesture_type': result.gesture_type.value,
            'confidence': result.confidence,
            'description': result.description,
            'hands_detected': result.hand_landmarks is not None
        }

# Global instance
_gesture_recognizer = None

def get_gesture_recognizer() -> WorkingGestureRecognizer:
    """Get or create the global gesture recognizer instance"""
    global _gesture_recognizer
    if _gesture_recognizer is None:
        _gesture_recognizer = WorkingGestureRecognizer()
    return _gesture_recognizer

# Convenience functions for backward compatibility
def recognize_gesture(base64_image: str) -> GestureResult:
    """Recognize gesture from base64 encoded image input"""
    recognizer = get_gesture_recognizer()
    return recognizer.recognize_gesture(base64_image)

def detect_emergency_gestures(image_input) -> Dict[str, Any]:
    """Detect emergency gestures from image input"""
    recognizer = get_gesture_recognizer()
    return recognizer.detect_emergency_gestures(image_input)

if __name__ == "__main__":
    # Test the gesture recognizer
    recognizer = WorkingGestureRecognizer()
    logger.info("Gesture recognition system ready for testing")
