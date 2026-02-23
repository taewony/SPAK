import random

def generate_addition_example_reversed(n1, n2):
    ans = n1 + n2
    
    # Format: "321+654=975" (Everything is reversed!)
    # n1=123 -> 321, n2=456 -> 654, ans=579 -> 975
    s1_rev = str(n1)[::-1]
    s2_rev = str(n2)[::-1]
    ans_rev = str(ans)[::-1]
    
    return f"{s1_rev}+{s2_rev}={ans_rev}"

def test_logic():
    print("--- Double Reverse Logic Verification ---")
    
    test_cases = [
        (123, 456, "321+654=975"),   # Simple case: 123+456=579
        (9, 1, "9+1=01"),             # Carry case: 9+1=10
        (99, 1, "99+1=001"),          # Double carry: 99+1=100
        (1234, 5678, "4321+8765=2196") # Complex: 1234+5678=6912
    ]
    
    all_pass = True
    for n1, n2, expected in test_cases:
        result = generate_addition_example_reversed(n1, n2)
        status = "✅ PASS" if result == expected else f"❌ FAIL (Got: {result})"
        print(f"Test {n1} + {n2}: Expected '{expected}' -> {status}")
        if result != expected: all_pass = False
        
    if all_pass:
        print("\nALL LOGIC TESTS PASSED!")
        print("The 'Double Reverse' implementation correctly aligns digits for the Transformer.")
    else:
        print("\nLOGIC TEST FAILED. Please check the implementation.")

if __name__ == "__main__":
    test_logic()
