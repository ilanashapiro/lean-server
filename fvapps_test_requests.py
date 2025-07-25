# just copied the spect from 0001
curl --request POST \
  --url http://localhost:8008/check_problem_solution \
  --header 'Content-Type: application/json' \
  --data '{
		"problem_id": "0001",
		"solution": "def solve_max_diagonal_move (n m k : Int) : Int :=\n  sorry\n\ndef abs (n : Int) : Int :=\n  if n ≥ 0 then n else -n\n\ntheorem result_bound (n m k : Int) (h: -1000 <= n ∧ n <= 1000) (h2: -1000 <= m ∧ m <= 1000) (h3: 0 <= k ∧ k <= 2000) :\n  let r := solve_max_diagonal_moves n m k\n  r = -1 ∨ r ≤ k := sorry\n\ntheorem result_parity (n m k : Int) (h: -1000 <= n ∧ n <= 1000) (h2: -1000 <= m ∧ m <= 1000) (h3: 0 <= k ∧ k <= 2000) :\n  let r := solve_max_diagonal_moves n m k\n  let max_dist := max (abs n) (abs m)\n  r ≠ -1 → (r % 2 = max_dist % 2 ∨ r % 2 = (max_dist - 1) % 2) := sorry\n\ntheorem insufficient_moves (n : Int) (h: 1 <= n ∧ n <= 1000) :\n  let k := abs n - 1\n  solve_max_diagonal_moves n n k = -1 := sorry\n\ntheorem symmetry (n m : Int) (h: -1000 <= n ∧ n <= 1000) (h2: -1000 <= m ∧ m <= 1000) :\n  let k := max (abs n) (abs m) * 2\n  let r1 := solve_max_diagonal_moves n m k\n  let r2 := solve_max_diagonal_moves (-n) m k\n  let r3 := solve_max_diagonal_moves n (-m) k\n  let r4 := solve_max_diagonal_moves (-n) (-m) k\n  r1 = r2 ∧ r2 = r3 ∧ r3 = r4 := sorry"
}' | jq


# model generated response for 0967
curl --request POST \
  --url http://localhost:8008/check_problem_solution \
  --header 'Content-Type: application/json' \
  --data '{
		"problem_id": "0967",
		"solution": "import List\ndef countsubsetsum (target : Nat) (arr : List Nat) : Nat :=\n  match arr with\n  | [] => if target = 0 then 1 else 0\n  | x::xs => countsubsetsum target xs + countsubsetsum (target - x) xs\n\ndef List.sum : List Nat → Nat\n  | [] => 0\n  | x::xs => x + List.sum xs\n\ntheorem zero_sum_always_has_one_solution {arr : List Nat} :\n  countsubsetsum 0 arr = 1 := by\n  induction arr\n  · simp\n  · simp [ih]\n\ntheorem single_element_sums {arr : List Nat} {x : Nat} :\n  x ∈ arr → countsubsetsum x arr ≥ 1 := by\n  induction arr\n  · simp\n  · intro h\n    simp [ih]\n    cases h\n    · simp\n    · simp [ih]\n\ntheorem results_non_negative {target : Nat} {arr : List Nat} :\n  countsubsetsum target arr ≥ 0 := by\n  induction arr\n  · simp\n  · simp [ih]"
}' | jq

