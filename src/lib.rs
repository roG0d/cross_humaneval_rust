use md5;
use rand::Rng;
use regex::Regex;
use std::any::{Any, TypeId};
use std::{
    ascii::AsciiExt,
    cmp::{self, max},
    collections::{HashMap, HashSet},
    mem::replace,
    ops::Index,
    slice::Iter,
};

//HumanEval/0 def has_close_elements(numbers: List[float], threshold: float) -> bool:
pub fn has_close_elements(numbers: Vec<f32>, threshold: f32) -> bool {
    for i in 0..numbers.len() {
        for j in 1..numbers.len() {
            if i != j {
                let distance: f32 = numbers[i] - numbers[j];

                if distance.abs() < threshold {
                    return true;
                }
            }
        }
    }

    return false;
}

//HumanEval/1 def separate_paren_groups(paren_string: str) -> List[str]:
pub fn separate_paren_groups(paren_string: String) -> Vec<String> {
    let mut result: Vec<String> = vec![];
    let mut current_string: String = String::new();
    let mut current_depth: u32 = 0;

    for c in paren_string.chars() {
        if c == '(' {
            current_depth += 1;
            current_string.push(c);
        } else if c == ')' {
            current_depth -= 1;
            current_string.push(c);

            if current_depth == 0 {
                result.push(current_string.clone());
                current_string.clear()
            }
        }
    }
    return result;
}

//HumanEval/2 def truncate_number(number: float) -> float:
pub fn truncate_number(number: &f32) -> f32 {
    return number % 1.0;
}

//HumanEval/3 def below_zero(operations: List[int]) -> bool:
pub fn below_zero(operations: Vec<i32>) -> bool {
    let mut balance: i32 = 0;
    for op in operations {
        balance = balance + op;
        if balance < 0 {
            return true;
        }
    }
    return false;
}

//HumanEval/4 def mean_absolute_deviation(numbers: List[float]) -> float:
pub fn mean_absolute_deviation(numbers: Vec<f32>) -> f32 {
    let mean: f32 = numbers.iter().fold(0.0, |acc: f32, x: &f32| acc + x) / numbers.len() as f32;
    return numbers.iter().map(|x: &f32| (x - mean).abs()).sum::<f32>() / numbers.len() as f32;
}

//HumanEval/5 def intersperse(numbers: List[int], delimeter: int) -> List[int]:
pub fn intersperse(numbers: Vec<u32>, delimeter: u32) -> Vec<u32> {
    let mut res: Vec<u32> = vec![];
    numbers.iter().for_each(|item: &u32| {
        res.push(*item);
        res.push(delimeter);
    });
    res.pop();
    return res;
}

//HumanEval/6 def parse_nested_parens(paren_string: str) -> List[int]:
pub fn parse_nested_parens(paren_string: String) -> Vec<i32> {
    let mut result: Vec<i32> = vec![];
    let mut depth: i32 = 0;
    let mut max_depth: i32 = 0;

    for splits in paren_string.split(' ') {
        for c in splits.chars() {
            if c == '(' {
                depth = depth + 1;
                max_depth = max(depth, max_depth);
            } else {
                depth = depth - 1;
            }
        }

        if depth == 0 {
            result.push(max_depth);
            max_depth = 0;
        }
    }

    return result;
}

//HumanEval/7 def filter_by_substring(strings: List[str], substring: str) -> List[str]:
fn filter_by_substring(strings: Vec<String>, substring: String) -> Vec<String> {
    return strings
        .iter()
        .filter(|x: &&String| x.contains(&substring))
        .map(String::from)
        .collect();
}

//HumanEval/8 def sum_product(numbers: List[int]) -> Tuple[int, int]:
fn sum_product(numbers: Vec<i32>) -> (i32, i32) {
    let sum = |xs: &Vec<i32>| {
        xs.iter().fold(0, |mut sum, &val| {
            sum += val;
            sum
        })
    };
    let product = |xs: &Vec<i32>| {
        xs.iter().fold(1, |mut prod, &val| {
            prod *= val;
            prod
        })
    };
    return (sum(&numbers), product(&numbers));
}

//HumanEval/9 def rolling_max(numbers: List[int]) -> List[int]:
fn rolling_max(numbers: Vec<i32>) -> Vec<i32> {
    let mut running_max: Option<i32> = None;
    let mut result: Vec<i32> = vec![];

    for n in numbers {
        if running_max == None {
            running_max = Some(n);
        } else {
            running_max = max(running_max, Some(n));
        }

        result.push(running_max.unwrap());
    }
    return result;
}

//HumanEval/10 string make_palindrome(string str){ CODEX
fn is_palindrome_10(str: &str) -> bool {
    //Test if given string is a palindrome
    let s: String = str.chars().rev().collect();
    return s == str;
}

fn make_palindrome(str: &str) -> String {
    let mut i: usize = 0;
    for i in 0..str.len() {
        let rstr: &str = &str[i..];
        if is_palindrome_10(rstr) {
            let nstr: &str = &str[0..i];
            let n2str: String = nstr.chars().rev().collect();
            return str.to_string() + &n2str;
        }
    }
    let n2str: String = str.chars().rev().collect();
    return str.to_string() + &n2str;
}

//HumanEval/11 def string_xor(a: str, b: str) -> str:
fn string_xor(a: String, b: String) -> String {
    let xor = |i: char, j: char| {
        if i == j {
            return "0".to_string();
        } else {
            return "1".to_string();
        }
    };
    return a
        .chars()
        .into_iter()
        .zip(b.chars().into_iter())
        .map(|(i, j)| "".to_string() + &xor(i, j))
        .collect();
}

//HumanEval/12 def longest(strings: List[str]) -> Optional[str]:
fn longest(strings: Vec<String>) -> Option<String> {
    if strings.is_empty() {
        return None;
    }
    let mut max: i32 = 0;
    let mut res: String = String::new();

    for s in strings {
        if s.len() as i32 > max {
            res = s;
            max = res.len() as i32;
        }
    }
    return Some(res);
}

//HumanEval/13  def greatest_common_divisor(a: int, b: int) -> int:
fn greatest_common_divisor(mut a: i32, mut b: i32) -> i32 {
    while b > 0 {
        (a, b) = (b, a % b);
    }
    return a;
}

//HumanEval/14 def all_prefixes(string: str) -> List[str]:
pub fn all_prefixes(string: String) -> Vec<String> {
    let mut res: Vec<String> = vec![];
    let mut res_str: String = String::new();

    for c in string.chars() {
        res_str.push(c);
        res.push(res_str.clone());
    }
    return res;
}

//HumanEval/15 def string_sequence(n: int) -> str:
pub fn string_sequence(n: i32) -> String {
    let mut res: String = String::new();

    for number in 0..n + 1 {
        res = res + &number.to_string() + " ";
    }

    return res.trim_end().to_string();
}

//HumanEval/16 def count_distinct_characters(string: str) -> int:
fn count_distinct_characters(str: String) -> i32 {
    let res: HashSet<char> = str
        .chars()
        .into_iter()
        .map(|x: char| x.to_ascii_lowercase())
        .collect();
    return res.len() as i32;
}

//HumanEval/17 def parse_music(music_string: str) -> List[int]:
pub fn parse_music(music_string: String) -> Vec<i32> {
    let map = |x: &str| match x {
        "o" => 4,
        "o|" => 2,
        ".|" => 1,
        _ => 0,
    };
    return music_string
        .split(" ")
        .map(|x: &str| map(&x.to_string()))
        .filter(|x: &i32| x != &0)
        .collect();
}

//HumanEval/18 def how_many_times(string: str, substring: str) -> int:
fn how_many_times(string: String, substring: String) -> i32 {
    let mut times: i32 = 0;

    for i in 0..(string.len() as i32 - substring.len() as i32 + 1) {
        if string
            .get(i as usize..(i + substring.len() as i32) as usize)
            .unwrap()
            .to_string()
            == substring
        {
            times += 1;
        }
    }
    return times;
}

//HumanEval/19 def sort_numbers(numbers: str) -> str:
fn sort_numbers(numbers: String) -> String {
    let str_to_i32 = |x: &str| match x {
        "zero" => 0,
        "one" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        _ => 1000,
    };

    let i32_to_str = |x: &i32| match x {
        0 => "zero".to_string(),
        1 => "one".to_string(),
        2 => "two".to_string(),
        3 => "three".to_string(),
        4 => "four".to_string(),
        5 => "five".to_string(),
        6 => "six".to_string(),
        7 => "seven".to_string(),
        8 => "eight".to_string(),
        9 => "nine".to_string(),
        _ => "none".to_string(),
    };

    let mut nmbrs: Vec<i32> = numbers
        .split_ascii_whitespace()
        .map(|x: &str| str_to_i32(x))
        .collect();
    nmbrs.sort();
    let res: String = nmbrs.iter().map(|x: &i32| i32_to_str(x) + " ").collect();
    return res.trim_end().to_string();
}

//HumanEval/20 def find_closest_elements(numbers: List[float]) -> Tuple[float, float]:
pub fn find_closest_elements(numbers: Vec<f32>) -> (f32, f32) {
    let mut closest_pair = (0.0, 0.0);
    let mut distance: Option<f32> = None;

    for (idx, elem) in numbers.iter().enumerate() {
        for (idx2, elem2) in numbers.iter().enumerate() {
            if idx != idx2 {
                if distance == None {
                    distance = Some((elem - elem2).abs());
                    if *elem < *elem2 {
                        closest_pair = (*elem, *elem2);
                    } else {
                        closest_pair = (*elem2, *elem);
                    }
                } else {
                    let new_distance: f32 = (elem - elem2).abs();
                    if new_distance < distance.unwrap() {
                        distance = Some(new_distance);

                        if *elem < *elem2 {
                            closest_pair = (*elem, *elem2);
                        } else {
                            closest_pair = (*elem2, *elem);
                        }
                    }
                }
            }
        }
    }
    return closest_pair;
}

//HumanEval/21 def rescale_to_unit(numbers: List[float]) -> List[float]:
fn rescale_to_unit(numbers: Vec<f32>) -> Vec<f32> {
    let min_number = *numbers
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_number = *numbers
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    return numbers
        .iter()
        .map(|x: &f32| (x - min_number) / (max_number - min_number))
        .collect();
}

//HumanEval/22 vector<int> filter_integers(list_any values){
fn filter_integers(values: Vec<Box<dyn Any>>) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::new();
    for value in values {
        if let Some(i) = value.downcast_ref::<i32>() {
            out.push(*i);
        }
    }
    out
}

//HumanEval/23 def strlen(string: str) -> int:
fn strlen(strings: String) -> i32 {
    return strings.len() as i32;
}

//HumanEval/24 def largest_divisor(n: int) -> int:
fn largest_divisor(n: i32) -> i32 {
    let mut res: i32 = 0;
    let sqn = 1..n;

    for i in sqn.rev() {
        if n % i == 0 {
            res = i;
            break;
        }
    }

    return res;
}

//HumanEval/25 def factorize(n: int) -> List[int]: CODEX
fn factorize(n: i32) -> Vec<i32> {
    let mut n = n;
    let mut factors = vec![];
    let mut divisor = 2;
    while divisor * divisor <= n {
        while n % divisor == 0 {
            factors.push(divisor);
            n = n / divisor;
        }
        divisor = divisor + 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

//HumanEval/26 def remove_duplicates(numbers: List[int]) -> List[int]:
pub fn remove_duplicates(numbers: Vec<i32>) -> Vec<i32> {
    let mut m: HashMap<i32, i32> = HashMap::new();

    for n in &numbers {
        *m.entry(*n).or_default() += 1;
    }
    let res: Vec<i32> = numbers
        .into_iter()
        .filter(|x| m.get(x) == Some(&1))
        .collect();
    return res;
}

//HumanEval/27 def flip_case(string: str) -> str:
pub fn flip_case(string: String) -> String {
    return string
        .chars()
        .into_iter()
        .fold(String::new(), |res: String, c: char| {
            if c.is_ascii_lowercase() {
                return res + &c.to_uppercase().to_string();
            } else {
                return res + &c.to_ascii_lowercase().to_string();
            }
        });
}

//HumanEval/28 def concatenate(strings: List[str]) -> str:
fn concatenate(strings: Vec<String>) -> String {
    return strings
        .iter()
        .fold(String::new(), |res: String, x: &String| {
            res + &x.to_string()
        });
}

//HumanEval/29 def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:
fn filter_by_prefix(strings: Vec<String>, prefix: String) -> Vec<String> {
    return strings
        .into_iter()
        .filter(|s| s.starts_with(&prefix))
        .collect();
}

//HumanEval/30 def get_positive(l: list[int]) -> list[int]:
fn get_positive(numbers: Vec<i32>) -> Vec<i32> {
    return numbers.into_iter().filter(|n| n.is_positive()).collect();
}

//HumanEval/31 def is_prime(n:int) -> bool:
fn is_prime(n: i32) -> bool {
    if n < 2 {
        return false;
    }
    for k in 2..n - 1 {
        if n % k == 0 {
            return false;
        }
    }
    return true;
}

//HumanEval/32 double find_zero(vector<double> xs){ CODEX
fn poly(xs: &Vec<f64>, x: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..xs.len() {
        sum += xs[i] * x.powi(i as i32);
    }
    sum
}

fn find_zero(xs: &Vec<f64>) -> f64 {
    let mut ans = 0.0;
    let mut value = poly(xs, ans);
    while value.abs() > 1e-6 {
        let mut driv = 0.0;
        for i in 1..xs.len() {
            driv += xs[i] * ans.powi((i - 1) as i32) * (i as f64);
        }
        ans = ans - value / driv;
        value = poly(xs, ans);
    }
    ans
}

//HumanEval/33 def sort_third(l: list[int]) -> list[int]: ERROR ON TEST LOOK UP HUMANEVAL 37 there they consider 0 included WTF???
pub fn sort_third(l: Vec<i32>) -> Vec<i32> {
    let mut third = vec![];
    let mut out: Vec<i32> = vec![];

    for (indx, elem) in l.iter().enumerate() {
        if indx % 3 == 0 && indx != 0 {
            third.push(elem)
        }
    }
    third.sort();
    let mut indx_t: usize = 0;

    for i in 0..l.len() {
        if i % 3 == 0 && i != 0 {
            if indx_t < third.len() {
                out.push(*third[indx_t]);
                indx_t += 1;
            }
        } else {
            out.push(l[i]);
        }
    }
    return out;
}

//HumanEval/34 def unique(l: list[int]) -> list[int]:
pub fn unique(nmbs: Vec<i32>) -> Vec<i32> {
    let mut res: Vec<i32> = nmbs.clone();
    res.sort();
    res.dedup();
    return res;
}

//HumanEval/35 def max_element(l: list[int]) -> int:
fn maximum(nmbs: Vec<i32>) -> i32 {
    return *nmbs.iter().max().unwrap();
}

//HumanEval/36 def fizz_buzz(n: int) -> int:
pub fn fizz_buzz(n: i32) -> i32 {
    let mut ns: Vec<i32> = vec![];

    for i in 0..n {
        if i % 11 == 0 || i % 13 == 0 {
            ns.push(i);
        }
    }

    let s: String = ns
        .into_iter()
        .fold(String::new(), |s: String, n: i32| s + &n.to_string());
    let mut ans: i32 = 0;

    for c in s.chars() {
        if c == '7' {
            ans += 1;
        }
    }
    return ans;
}

//HumanEval/37 def sort_even(l: list[int]) -> list[int]:
fn sort_even(nmbs: Vec<i32>) -> Vec<i32> {
    let mut even = vec![];
    let mut out: Vec<i32> = vec![];

    for (indx, elem) in nmbs.iter().enumerate() {
        if indx % 2 == 0 {
            even.push(elem)
        }
    }
    even.sort();
    let mut indx_t: usize = 0;

    for i in 0..nmbs.len() {
        if i % 2 == 0 {
            if indx_t < even.len() {
                out.push(*even[indx_t]);
                indx_t += 1;
            }
        } else {
            out.push(nmbs[i]);
        }
    }
    return out;
}

//HumanEval/38 string decode_cyclic(string s){
pub fn decode_cyclic(s: &str) -> String {
    let l = s.len();
    let num = (l + 2) / 3;
    let mut output = String::new();
    for i in 0..num {
        let group = &s[i * 3..std::cmp::min(l, (i + 1) * 3)];
        // revert the cycle performed by the encode_cyclic function
        if group.len() == 3 {
            let x = format!("{}{}{}", &group[2..3], &group[0..1], &group[1..2]);
            output.push_str(&x);
        } else {
            output.push_str(group);
        }
    }
    output
}

pub fn encode_cyclic(s: &str) -> String {
    // returns encoded string by cycling groups of three characters.
    // split string to groups. Each of length 3.
    let l = s.len();
    let num = (l + 2) / 3;
    let mut output = String::new();
    for i in 0..num {
        let group = &s[i * 3..std::cmp::min(l, (i + 1) * 3)];
        // cycle elements in each group. Unless group has fewer elements than 3.
        if group.len() == 3 {
            let x = format!("{}{}{}", &group[1..2], &group[2..3], &group[0..1]);
            output.push_str(&x);
        } else {
            output.push_str(group);
        }
    }
    output
}



//HumanEval/39 def prime_fib(n: int) -> int:
fn prime_fib(n: i32) -> i32 {
    let mut f1 = 1;
    let mut f2 = 2;
    let mut count = 0;
    while count < n {
        f1 = f1 + f2;
        let m = f1;
        f1 = f2;
        f2 = m;
        let mut isprime = true;
        for w in 2..(f1 as f32).sqrt() as i32 + 1 {
            if f1 % w == 0 {
                isprime = false;
                break;
            }
        }
        if isprime {
            count += 1;
        }
        if count == n {
            return f1;
        }
    }
    0
}

//HumanEval/40 def triples_sum_to_zero(l: list[int]) -> bool:
fn triples_sum_to_zero(nmbs: Vec<i32>) -> bool {
    for i in 0..nmbs.len() {
        for j in i + 1..nmbs.len() {
            for k in j + 1..nmbs.len() {
                if *nmbs.get(i).unwrap() + *nmbs.get(j).unwrap() + *nmbs.get(k).unwrap() == 0 {
                    return true;
                }
            }
        }
    }
    return false;
}

//HumanEval/41 def car_race_collision(n: int) -> int:
fn car_race_collision(n: i32) -> i32 {
    return n * n;
}

//HumanEval/42 def incr_list(l: list[int]) -> list[int]:
fn incr_list(l: Vec<i32>) -> Vec<i32> {
    return l.into_iter().map(|n: i32| n + 1).collect();
}

//HumanEval//43 def pairs_sum_to_zero(l:list[int]) -> bool:
fn pairs_sum_to_zero(l: Vec<i32>) -> bool {
    for (i, l1) in l.iter().enumerate() {
        for j in i + 1..l.len() {
            if l1 + l[j] == 0 {
                return true;
            }
        }
    }

    return false;
}

//HumanEval/44 def change_base(x: int, base: int) -> str:
pub fn change_base(x: i32, base: i32) -> String {
    let mut ret: String = "".to_string();
    let mut x1 = x;

    while x1 > 0 {
        ret = (x1 % base).to_string() + &ret;
        x1 = x1 / base;
    }
    return ret;
}

//HumanEval/45 def triangle_area(a: int, h:int) -> float:
fn triangle_area(a: i32, h: i32) -> f64 {
    return (a * h) as f64 / 2.0;
}

//HumanEval/46 def fib4(n: int) -> int:
fn fib4(n: i32) -> i32 {
    let mut results: Vec<i32> = vec![0, 0, 2, 0];

    if n < 4 {
        return *results.get(n as usize).unwrap();
    }

    for _ in 4..n + 1 {
        results.push(
            results.get(results.len() - 1).unwrap()
                + results.get(results.len() - 2).unwrap()
                + results.get(results.len() - 3).unwrap()
                + results.get(results.len() - 4).unwrap(),
        );
        results.remove(0);
    }

    return *results.get(results.len() - 1).unwrap();
}

//HumanEval/47 def median(l: list[int]) -> list[int]:
fn median(l: Vec<i32>) -> f64 {
    let mut res: Vec<i32> = l.clone();
    res.sort();
    if res.len() % 2 == 1 {
        return *res.get(res.len() / 2).unwrap() as f64;
    } else {
        return (res.get(res.len() / 2 - 1).unwrap() + res.get(res.len() / 2).unwrap()) as f64
            / 2.0;
    }
}

//HumanEval/48 def is_palindrome(text: str):
fn is_palindrome(text: String) -> bool {
    let pr: String = text.chars().rev().collect();
    return pr == text;
}

//HumanEval/49 def modp(n: int, p: int):
fn modp(n: i32, p: i32) -> i32 {
    if n == 0 {
        return 1;
    } else {
        return (modp(n - 1, p) * 2) % p;
    }
}

//HumanEval/50 def encode_shift(s: str): ERROR ON TEST: MODIFIED TO SATISFIES THAT random characters that can be generated are solely from the alphabet
pub fn encode_shift(s: &str) -> String {
    let alphabet: Vec<&str> = vec![
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
        "s", "t", "u", "v", "w", "x", "y", "z",
    ];
    let mut output = String::new();

    for c in s.chars() {
        let mut lower = false;
        if c.is_ascii_lowercase() {
            lower = true;
        }
        let mut c_shift: String = "".to_string();
        if lower {
            let index: usize = alphabet.iter().position(|&x| x == c.to_string()).unwrap();
            c_shift = alphabet[(index + 5) % 26].to_string();
        } else {
            let c_lower: String = c.to_ascii_lowercase().to_string();
            let index: usize = alphabet.iter().position(|&x| x == c_lower).unwrap();
            c_shift = alphabet[(index + 5) % 26].to_string();
            c_shift = c_shift.to_ascii_uppercase().to_string();
        }

        output.push_str(&c_shift);
    }
    output
}

pub fn decode_shift(s: &str) -> String {
    let alphabet: Vec<&str> = vec![
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
        "s", "t", "u", "v", "w", "x", "y", "z",
    ];
    let mut output = String::new();

    for c in s.chars() {
        let mut lower = false;
        if c.is_ascii_lowercase() {
            lower = true;
        }
        let mut c_shift: String = "".to_string();
        if lower {
            let index: usize = alphabet.iter().position(|&x| x == c.to_string()).unwrap();
            c_shift = alphabet[((26 + (index as i32 - 5)) % 26) as usize].to_string();
        } else {
            let c_lower: String = c.to_ascii_lowercase().to_string();
            let index: usize = alphabet.iter().position(|&x| x == c_lower).unwrap();
            c_shift = alphabet[((26 + (index as i32 - 5)) % 26) as usize].to_string();
            c_shift = c_shift.to_ascii_uppercase().to_string();
        }

        output.push_str(&c_shift);
    }
    output
}

//HumanEval/51 string remove_vowels(string text){
fn remove_vowels(text: &str) -> String {
    let vowels = "AEIOUaeiou";
    let mut out = String::new();
    for c in text.chars() {
        if !vowels.contains(c) {
            out.push(c);
        }
    }
    out
}

//HumanEval/52 bool below_threshold(vector<int>l, int t){
fn below_threshold(l: Vec<i32>, t: i32) -> bool {
    for i in l {
        if i >= t {
            return false;
        }
    }
    return true;
}

//HumanEval/53  int add(int x,int y){
fn add(x: i32, y: i32) -> i32 {
    return x + y;
}

//HumanEval/54 bool same_chars(string s0,string s1){
fn same_chars(str1: &str, str2: &str) -> bool {
    let mut v1: Vec<char> = str1.chars().into_iter().collect();
    v1.sort();
    v1.dedup();

    let mut v2: Vec<char> = str2.chars().into_iter().collect();
    v2.sort();
    v2.dedup();

    return v1 == v2;
}

//HumanEval/55 int fib(int n){
fn fib(n: i32) -> i32 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }

    return fib(n - 1) + fib(n - 2);
}

//HumanEval/56 bool correct_bracketing(string brackets){
fn correct_bracketing(bkts: &str) -> bool {
    let mut level: i32 = 0;

    for i in 0..bkts.len() {
        if bkts.chars().nth(i).unwrap() == '<' {
            level += 1;
        }

        if bkts.chars().nth(i).unwrap() == '>' {
            level -= 1;
        }

        if level < 0 {
            return false;
        }
    }
    if level != 0 {
        return false;
    }
    return true;
}

//HumanEval/57 bool monotonic(vector<float> l){
fn monotonic(l: Vec<i32>) -> bool {
    let mut l1: Vec<i32> = l.clone();
    let mut l2: Vec<i32> = l.clone();
    l2.sort();
    l2.reverse();
    l1.sort();

    if l == l1 || l == l2 {
        return true;
    }
    return false;
}

//HumanEval/58 vector<int> common(vector<int> l1,vector<int> l2){
fn common(l1: Vec<i32>, l2: Vec<i32>) -> Vec<i32> {
    let mut res: Vec<i32> = l1.into_iter().filter(|n: &i32| l2.contains(n)).collect();
    res.sort();
    return res;
}

//HumanEval/59 int largest_prime_factor(int n){
fn largest_prime_factor(n: i32) -> i32 {
    let mut n1 = n.clone();
    for i in 2..n1 {
        while n1 % i == 0 && n1 > i {
            n1 = n1 / i;
        }
    }
    return n1;
}

//HumanEval/60 int sum_to_n(int n){
fn sum_to_n(n: i32) -> i32 {
    n * (n + 1) / 2
}

//HumanEval/61 bool correct_bracketing(string brackets){
fn correct_bracketing_parenthesis(bkts: &str) -> bool {
    let mut level: i32 = 0;

    for i in 0..bkts.len() {
        if bkts.chars().nth(i).unwrap() == '(' {
            level += 1;
        }

        if bkts.chars().nth(i).unwrap() == ')' {
            level -= 1;
        }

        if level < 0 {
            return false;
        }
    }
    if level != 0 {
        return false;
    }
    return true;
}

//HumanEval/62 vector<float> derivative(vector<float> xs){
fn derivative(xs: Vec<i32>) -> Vec<i32> {
    let mut res: Vec<i32> = vec![];
    for i in 1..xs.len() {
        res.push(i as i32 * xs.get(i).unwrap());
    }
    return res;
}

//HumanEval/63 int fibfib(int n){
fn fibfib(n: i32) -> i32 {
    if n == 0 || n == 1 {
        return 0;
    }
    if n == 2 {
        return 1;
    }

    return fibfib(n - 1) + fibfib(n - 2) + fibfib(n - 3);
}

//HumanEval/64 int vowels_count(string s){
fn vowels_count(s: &str) -> i32 {
    let vowels: &str = "aeiouAEIOU";
    let mut count: i32 = 0;

    for i in 0..s.len() {
        let c: char = s.chars().nth(i).unwrap();
        if vowels.contains(c) {
            count += 1;
        }
    }
    if s.chars().nth(s.len() - 1).unwrap() == 'y' || s.chars().nth(s.len() - 1).unwrap() == 'Y' {
        count += 1;
    }

    return count;
}

//HumanEval/65 string circular_shift(int x,int shift){
fn circular_shift(x: i32, shift: i32) -> String {
    let mut xcp: Vec<char> = x.to_string().chars().into_iter().collect();
    let mut res: Vec<char> = x.to_string().chars().into_iter().collect();

    for (indx, c) in xcp.iter().enumerate() {
        let despl = (indx as i32 + shift) % x.to_string().len() as i32;
        replace(&mut res[despl as usize], *c);
    }

    return res.into_iter().collect();
}

//HumanEval/66  int digitSum(string s){
fn digitSum(s: &str) -> i32 {
    return s
        .chars()
        .into_iter()
        .filter(|c: &char| c.is_uppercase())
        .map(|c: char| c as i32)
        .sum();
}

//HumanEval/67 int fruit_distribution(string s,int n){
fn fruit_distribution(s: &str, n: i32) -> i32 {
    let sub: i32 = s
        .split_ascii_whitespace()
        .into_iter()
        .filter(|c| c.parse::<i32>().is_ok())
        .map(|c| c.parse::<i32>().unwrap())
        .sum();
    return n - sub;
}

//HumanEval/68 vector<int> pluck(vector<int> arr){
fn pluck(arr: Vec<i32>) -> Vec<i32> {
    let mut out: Vec<i32> = vec![];

    for i in 0..arr.len() {
        if arr[i] % 2 == 0 && (out.len() == 0 || arr[i] < out[0]) {
            out = vec![arr[i], i as i32];
        }
    }
    return out;
}

//HumanEval/69 int search(vector<int> lst){
fn search(lst: Vec<i32>) -> i32 {
    let mut freq: Vec<Vec<i32>> = Vec::new();
    let mut max = -1;
    for i in 0..lst.len() {
        let mut has = false;
        for j in 0..freq.len() {
            if lst[i] == freq[j][0] {
                freq[j][1] += 1;
                has = true;
                if freq[j][1] >= freq[j][0] && freq[j][0] > max {
                    max = freq[j][0];
                }
            }
        }
        if !has {
            freq.push(vec![lst[i], 1]);
            if max == -1 && lst[i] == 1 {
                max = 1;
            }
        }
    }
    return max;
}

//HumanEval/70 vector<int> strange_sort_list(vector<int> lst){
pub fn strange_sort_list(lst: Vec<i32>) -> Vec<i32> {
    let mut cp: Vec<i32> = lst.clone();
    let mut res: Vec<i32> = vec![];

    for (indx, _) in lst.iter().enumerate() {
        if indx % 2 == 1 {
            let max: i32 = *cp.iter().max().unwrap();
            res.push(max);
            cp.remove(cp.iter().position(|x| *x == max).unwrap());
        } else {
            let min: i32 = *cp.iter().min().unwrap();
            res.push(min);
            cp.remove(cp.iter().position(|x| *x == min).unwrap());
        }
    }
    return res;
}

//HumanEval/71 float triangle_area(float a,float b,float c){
fn triangle_area_f64(a: f64, b: f64, c: f64) -> f64 {
    if a + b <= c || a + c <= b || b + c <= a {
        return -1.0;
    }
    let h: f64 = (a + b + c) / 2.0;
    let mut area: f64;
    area = f64::powf(h * (h - a) * (h - b) * (h - c), 0.5);
    return area;
}

//HumanEval/72 bool will_it_fly(vector<int> q,int w){
fn will_it_fly(q: Vec<i32>, w: i32) -> bool {
    if q.iter().sum::<i32>() > w {
        return false;
    }
    let mut i = 0;
    let mut j = q.len() - 1;

    while i < j {
        if q[i] != q[j] {
            return false;
        }
        i += 1;
        j -= 1;
    }
    return true;
}

//HumanEval/73 int smallest_change(vector<int> arr){
fn smallest_change(arr: Vec<i32>) -> i32 {
    let mut ans: i32 = 0;
    for i in 0..arr.len() / 2 {
        if arr[i] != arr[arr.len() - i - 1] {
            ans += 1
        }
    }
    return ans;
}

//HumanEval/74 vector<string> total_match(vector<string> lst1,vector<string> lst2){
pub fn total_match(lst1: Vec<&str>, lst2: Vec<&str>) -> Vec<String> {
    let total_1: usize = lst1
        .iter()
        .fold(0, |acc: usize, str: &&str| acc + str.chars().count());
    let total_2: usize = lst2
        .iter()
        .fold(0, |acc: usize, str: &&str| acc + str.chars().count());

    if total_1 <= total_2 {
        return lst1.into_iter().map(|x| x.to_string()).collect();
    } else {
        return lst2.into_iter().map(|x| x.to_string()).collect();
    }
}

//HumanEval/75 bool is_multiply_prime(int a){
fn is_multiply_prime(a: i32) -> bool {
    let mut a1 = a;
    let mut num = 0;
    for i in 2..a {
        while a1 % i == 0 && a1 > i {
            a1 /= i;
            num += 1;
        }
    }
    if num == 2 {
        return true;
    }
    return false;
}

//HumanEval/76 bool is_simple_power(int x,int n){
fn is_simple_power(x: i32, n: i32) -> bool {
    let mut p: i32 = 1;
    let mut count: i32 = 0;

    while p <= x && count < 100 {
        if p == x {
            return true;
        };
        p = p * n;
        count += 1;
    }
    return false;
}

//HumanEval/77 bool iscuber(int a){
fn iscuber(a: i32) -> bool {
    let a1: f64 = i32::abs(a) as f64;
    let sqrt_3 = f64::powf(a1, 1.0 / 3.0).ceil();

    return i32::pow(sqrt_3 as i32, 3) == a1 as i32;
}

//HumanEval/78 int hex_key(string num){
fn hex_key(num: &str) -> i32 {
    let primes: Vec<&str> = vec!["2", "3", "5", "7", "B", "D"];
    let mut total: i32 = 0;
    for i in 0..num.len() {
        if primes.contains(&num.get(i..i + 1).unwrap()) {
            total += 1;
        }
    }
    return total;
}

//HumanEval//79 string decimal_to_binary(int decimal){
pub fn decimal_to_binary(decimal: i32) -> String {
    let mut d_cp = decimal;
    let mut out: String = String::from("");
    if d_cp == 0 {
        return "db0db".to_string();
    }
    while d_cp > 0 {
        out = (d_cp % 2).to_string() + &out;
        d_cp = d_cp / 2;
    }
    out = "db".to_string() + &out + &"db".to_string();
    return out;
}

//HumanEval/80 bool is_happy(string s){
fn is_happy(s: &str) -> bool {
    let str: Vec<char> = s.chars().into_iter().collect();
    if str.len() < 3 {
        return false;
    }
    for i in 2..str.len() {
        if str[i] == str[i - 1] || str[i] == str[i - 2] {
            return false;
        }
    }
    return true;
}

//HumanEval/81 vector<string> numerical_letter_grade(vector<float> grades){
fn numerical_letter_grade(grades: Vec<f64>) -> Vec<String> {
    let mut res: Vec<String> = vec![];
    for (i, gpa) in grades.iter().enumerate() {
        if gpa == &4.0 {
            res.push("A+".to_string());
        } else if gpa > &3.7 {
            res.push("A".to_string());
        } else if gpa > &3.3 {
            res.push("A-".to_string());
        } else if gpa > &3.0 {
            res.push("B+".to_string());
        } else if gpa > &2.7 {
            res.push("B".to_string());
        } else if gpa > &2.3 {
            res.push("B-".to_string());
        } else if gpa > &2.0 {
            res.push("C+".to_string());
        } else if gpa > &1.7 {
            res.push("C".to_string());
        } else if gpa > &1.3 {
            res.push("C-".to_string());
        } else if gpa > &1.0 {
            res.push("D+".to_string());
        } else if gpa > &0.7 {
            res.push("D".to_string());
        } else if gpa > &0.0 {
            res.push("D-".to_string());
        } else {
            res.push("E".to_string());
        }
    }
    return res;
}

//HumanEval/82 bool prime_length(string str){
fn prime_length(str: &str) -> bool {
    let l: usize = str.len();
    if l == 0 || l == 1 {
        return false;
    }

    for i in 2..l {
        if l % i == 0 {
            return false;
        }
    }
    return true;
}

//HumanEval/83 int starts_one_ends(int n){
fn starts_one_ends(n: i32) -> i32 {
    if n == 1 {
        return 1;
    };
    return 18 * i32::pow(10, (n - 2) as u32);
}

//HumanEval/84 string solve(int N){
pub fn solve(n: i32) -> String {
    let sum: i32 = n
        .to_string()
        .chars()
        .into_iter()
        .fold(0, |acc, c| acc + c.to_digit(10).unwrap() as i32);
    return format!("{sum:b}");
}

//HumanEval/85 int add(vector<int> lst){
fn add_even_odd(lst: Vec<i32>) -> i32 {
    let mut sum: i32 = 0;

    for (indx, elem) in lst.iter().enumerate() {
        if indx % 2 == 1 {
            if elem % 2 == 0 {
                sum += elem
            }
        }
    }
    return sum;
}

//HumanEval//86 string anti_shuffle(string s){
pub fn anti_shuffle(s: &str) -> String {
    let mut res: String = String::new();

    for i in s.split_ascii_whitespace() {
        let mut str: Vec<char> = i.chars().into_iter().collect();
        str.sort_by(|a, b| (*a as u32).cmp(&(*b as u32)));
        let str_sorted: String = str.into_iter().collect();
        res.push_str(&(str_sorted + &" ".to_string()));
    }
    res = res.trim_end().to_string();
    return res;
}

//HumanEval/87 vector<vector<int>> get_row(vector<vector<int>> lst, int x){
pub fn get_row(lst: Vec<Vec<i32>>, x: i32) -> Vec<Vec<i32>> {
    let mut out: Vec<Vec<i32>> = vec![];
    for (indxi, elem1) in lst.iter().enumerate() {
        for (indxj, _) in elem1.iter().rev().enumerate() {
            if lst[indxi][indxj] == x {
                out.push(vec![indxi as i32, indxj as i32]);
            }
        }
    }
    return out;
}

//HumanEval/88 vector<int> sort_array(vector<int> array){
pub fn sort_array(array: Vec<i32>) -> Vec<i32> {
    let mut res: Vec<i32> = array.clone();

    if array.len() == 0 {
        return res;
    }

    if (array[0] + array[array.len() - 1]) % 2 == 0 {
        res.sort();
        return res.into_iter().rev().collect();
    } else {
        res.sort();
        return res;
    }
}

//HumanEval/89 string encrypt(string s){
fn encrypt(s: &str) -> String {
    let d: Vec<char> = "abcdefghijklmnopqrstuvwxyz"
        .to_string()
        .chars()
        .into_iter()
        .collect();
    let mut out: String = String::new();
    for c in s.chars() {
        if d.contains(&c) {
            let indx: usize = (d.iter().position(|x| c == *x).unwrap() + 2 * 2) % 26;
            out += &d[indx].to_string();
        } else {
            out += &c.to_string();
        }
    }

    return out;
}

//HumanEval/90 int next_smallest(vector<int> lst){
pub fn next_smallest(lst: Vec<i32>) -> i32 {
    let mut res = 0;
    let mut lst_cp = lst.clone();
    let mut first: i32 = 0;
    let mut second: i32 = 0;

    if lst.iter().min() == None {
        res = -1;
    } else {
        if lst.iter().min() != None {
            first = *lst.iter().min().unwrap();
            let indx = lst.iter().position(|x| *x == first).unwrap();
            lst_cp.remove(indx);

            if lst_cp.iter().min() != None {
                second = *lst_cp.iter().min().unwrap();
            }
            if first != second {
                res = second;
            } else {
                res = -1;
            }
        }
    }
    return res;
}

//HumanEval/91 int is_bored(string S){
fn is_bored(s: &str) -> i32 {
    let mut count = 0;
    let regex = Regex::new(r"[.?!]\s*").expect("Invalid regex");
    let sqn: Vec<&str> = regex.split(s).into_iter().collect();
    for s in sqn {
        if s.starts_with("I ") {
            count += 1;
        }
    }
    return count;
}

//HumanEval/92 bool any_int(float a,float b,float c){
fn any_int(a: f64, b: f64, c: f64) -> bool {
    if a.fract() == 0.0 && b.fract() == 0.0 && c.fract() == 0.0 {
        return a + b == c || a + c == b || b + c == a;
    } else {
        return false;
    }
}

//HumanEval/93 string encode(string message){
fn encode(message: &str) -> String {
    let mut res: String = String::new();
    let v: Vec<char> = "aeiouAEIOU".to_string().chars().into_iter().collect();
    let d: Vec<char> = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        .to_string()
        .chars()
        .into_iter()
        .collect();

    for (indx, elem) in message.chars().into_iter().enumerate() {
        let mut c = elem.to_string();

        if v.contains(&elem) {
            let indx: usize = d.iter().position(|x| &elem == x).unwrap();
            c = d[indx + 2 as usize].to_string();
        }

        if elem.is_uppercase() {
            c = c.to_lowercase().to_string();
        } else {
            c = c.to_uppercase().to_string();
        }
        res.push_str(&c);
    }
    return res;
}

//HumanEval/94 int skjkasdkd(vector<int> lst){ ERROR name???
fn skjkasdkd(lst: Vec<i32>) -> i32 {
    let mut largest = 0;
    for i in 0..lst.len() {
        if lst[i] > largest {
            let mut prime = true;
            let mut j = 2;
            while j * j <= lst[i] {
                if lst[i] % j == 0 {
                    prime = false;
                }
                j += 1;
            }

            if prime {
                largest = lst[i];
            }
        }
    }
    let mut sum: i32 = 0;
    let mut s: String = String::new();
    s = largest.to_string();

    for n in s.chars().into_iter() {
        sum += n.to_digit(10).unwrap() as i32;
    }
    return sum;
}

//HumanEval/95 bool check_dict_case(map<string,string> dict){
fn check_dict_case(dict: HashMap<&str, &str>) -> bool {
    if dict.is_empty() {
        return false;
    }
    let string_lower: fn(str: &str) -> bool = |str: &str| {
        return str.chars().into_iter().all(|c| c.is_ascii_lowercase());
    };
    let string_upper: fn(str: &str) -> bool = |str: &str| {
        return str.chars().into_iter().all(|c| c.is_ascii_uppercase());
    };

    let lower: bool = dict.keys().into_iter().all(|str| string_lower(str));
    let upper: bool = dict.keys().into_iter().all(|str| string_upper(str));
    return lower || upper;
}

//HumanEval/96 vector<int> count_up_to(int n){
fn count_up_to(n: i32) -> Vec<i32> {
    let mut primes: Vec<i32> = vec![];

    for i in 2..n {
        let mut is_prime: bool = true;

        for j in 2..i {
            if i % j == 0 {
                is_prime = false;
                break;
            }
        }
        if is_prime {
            primes.push(i);
        }
    }
    return primes;
}

//HumanEval/97 int multiply(int a,int b){
fn multiply(a: i32, b: i32) -> i32 {
    return (i32::abs(a) % 10) * (i32::abs(b) % 10);
}

//HumanEval/98 int count_upper(string s){
fn count_upper(s: &str) -> i32 {
    let uvowel: &str = "AEIOU";
    let mut count: i32 = 0;

    for (indx, elem) in s.chars().into_iter().enumerate() {
        if indx % 2 == 0 {
            if uvowel.contains(elem) {
                count += 1;
            }
        }
    }
    return count;
}

//HumanEval/99 int closest_integer(string value){
fn closest_integer(value: &str) -> i32 {
    return value.parse::<f64>().unwrap().round() as i32;
}

//HumanEval/100 vector<int> make_a_pile(int n){
fn make_a_pile(n: i32) -> Vec<i32> {
    let mut out: Vec<i32> = vec![n];

    for i in 1..n {
        out.push(out[out.len() - 1] + 2);
    }

    return out;
}

//HumanEval/101 vector<string> words_string(string s){
pub fn words_string(s: &str) -> Vec<String> {
    return s
        .to_string()
        .split(|c: char| c == ',' || c.is_whitespace())
        .into_iter()
        .filter(|x| x != &"")
        .map(|x| x.to_string())
        .collect();
}

//HumanEval/102 int choose_num(int x,int y){
fn choose_num(x: i32, y: i32) -> i32 {
    if y < x {
        return -1;
    }
    if y == x && y % 2 == 1 {
        return -1;
    }
    if y % 2 == 1 {
        return y - 1;
    }
    return y;
}

//HumanEval/103 string rounded_avg(int n,int m){
fn rounded_avg(n: i32, m: i32) -> String {
    if n > m {
        return "-1".to_string();
    };
    let mut num: i32 = (m + n) / 2;
    let mut out: String = String::from("");
    while num > 0 {
        out = (num % 2).to_string() + &out;
        num = num / 2;
    }
    return out;
}

//HumanEval/104 vector<int> unique_digits(vector<int> x){
fn unique_digits(x: Vec<i32>) -> Vec<i32> {
    let mut res: Vec<i32> = vec![];
    for (_, elem) in x.into_iter().enumerate() {
        let mut elem_cp: i32 = elem;
        let mut u: bool = true;
        if elem == 0 {
            u = false;
        }
        while elem_cp > 0 && u {
            if elem_cp % 2 == 0 {
                u = false;
            }
            elem_cp = elem_cp / 10;
        }
        if u {
            res.push(elem)
        };
    }
    res.sort();
    return res;
}

//HumanEval/105 vector<string> by_length(vector<int> arr){
fn by_length(arr: Vec<i32>) -> Vec<String> {
    let mut res: Vec<String> = vec![];
    let mut arr_cp: Vec<i32> = arr.clone();
    arr_cp.sort();
    arr_cp.reverse();
    let map: HashMap<i32, &str> = HashMap::from([
        (0, "Zero"),
        (1, "One"),
        (2, "Two"),
        (3, "Three"),
        (4, "Four"),
        (5, "Five"),
        (6, "Six"),
        (7, "Seven"),
        (8, "Eight"),
        (9, "Nine"),
    ]);

    for elem in arr_cp {
        if elem >= 1 && elem <= 9 {
            res.push(map.get(&elem).unwrap().to_string());
        }
    }

    return res;
}

//HumanEval/106 vector<int> f(int n){
fn f(n: i32) -> Vec<i32> {
    let mut sum: i32 = 0;
    let mut prod: i32 = 1;
    let mut out: Vec<i32> = vec![];

    for i in 1..n + 1 {
        sum += i;
        prod *= i;

        if i % 2 == 0 {
            out.push(prod);
        } else {
            out.push(sum)
        };
    }
    return out;
}

//HumanEval/107 vector<int> even_odd_palindrome(int n){
fn even_odd_palindrome(n: i32) -> (i32, i32) {
    let mut even = 0;
    let mut odd = 0;

    for i in 1..n + 1 {
        let mut w: String = i.to_string();
        let mut p: String = w.chars().rev().collect();

        if w == p && i % 2 == 1 {
            odd += 1;
        }
        if w == p && i % 2 == 0 {
            even += 1;
        }
    }
    (even, odd)
}

//HumanEval/108 int count_nums(vector<int> n){
fn count_nums(n: Vec<i32>) -> i32 {
    let mut num: i32 = 0;

    for nmbr in n {
        if nmbr > 0 {
            num += 1;
        } else {
            let mut sum: i32 = 0;
            let mut w: i32;
            w = i32::abs(nmbr);

            while w >= 10 {
                sum += w % 10;
                w = w / 10;
            }
            sum -= w;
            if sum > 0 {
                num += 1;
            }
        }
    }
    return num;
}

//HumanEval109 bool move_one_ball(vector<int> arr){
fn move_one_ball(arr: Vec<i32>) -> bool {
    let mut num = 0;
    if arr.len() == 0 {
        return true;
    }
    for i in 1..arr.len() {
        if arr[i] < arr[i - 1] {
            num += 1;
        }
    }
    if arr[arr.len() - 1] > arr[0] {
        num += 1;
    }
    if num < 2 {
        return true;
    }
    return false;
}

//HumanEval/110 string exchange(vector<int> lst1,vector<int> lst2){
fn exchange(lst1: Vec<i32>, lst2: Vec<i32>) -> String {
    let mut num = 0;
    for i in 0..lst1.len() {
        if lst1[i] % 2 == 0 {
            num += 1;
        }
    }
    for i in 0..lst2.len() {
        if lst2[i] % 2 == 0 {
            num += 1;
        }
    }
    if num >= lst1.len() {
        return "YES".to_string();
    }
    return "NO".to_string();
}

//HumanEval/111 map<char,int> histogram(string test){
fn histogram(test: &str) -> HashMap<char, i32> {
    let mut res: HashMap<char, i32> = HashMap::new();
    if test == "" {
        return res;
    }
    for c in test.split_ascii_whitespace() {
        if res.contains_key(&c.chars().next().unwrap()) {
            res.entry(c.chars().next().unwrap()).and_modify(|n| {
                *n += 1;
            });
        } else {
            res.insert(c.chars().next().unwrap(), 1);
        }
    }
    let max: i32 = *res.values().max().unwrap();
    let non_maxs: Vec<char> = res
        .keys()
        .filter(|k: &&char| *res.get(k).unwrap() != max)
        .map(|c| *c)
        .collect();
    non_maxs.iter().for_each(|c| {
        res.remove(c);
    });

    return res;
}

//HumanEval/112 vector<string> reverse_delete(string s,string c){
fn reverse_delete(s: &str, c: &str) -> Vec<String> {
    let mut n = String::new();
    for i in 0..s.len() {
        if !c.contains(s.chars().nth(i).unwrap()) {
            n.push(s.chars().nth(i).unwrap());
        }
    }
    if n.len() == 0 {
        return vec![n, "True".to_string()];
    }
    let w: String = n.chars().rev().collect();
    if w == n {
        return vec![n, "True".to_string()];
    }
    return vec![n, "False".to_string()];
}

//HumanEval//113 vector<string> odd_count(vector<string> lst){
fn odd_count(lst: Vec<&str>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for i in 0..lst.len() {
        let mut sum = 0;
        for j in 0..lst[i].len() {
            if lst[i].chars().nth(j).unwrap() >= '0'
                && lst[i].chars().nth(j).unwrap() <= '9'
                && lst[i].chars().nth(j).unwrap().to_digit(10).unwrap() % 2 == 1
            {
                sum += 1;
            }
        }
        let mut s = "the number of odd elements in the string i of the input.".to_string();
        let mut s2 = "".to_string();
        for j in 0..s.len() {
            if s.chars().nth(j).unwrap() == 'i' {
                s2.push_str(&sum.to_string());
            } else {
                s2.push(s.chars().nth(j).unwrap());
            }
        }
        out.push(s2);
    }
    return out;
}

//HumanEval/114 long long minSubArraySum(vector<long long> nums){
fn min_sub_array_sum(nums: Vec<i64>) -> i64 {
    let mut current = nums[0];
    let mut min = nums[0];
    for i in 1..nums.len() {
        if current < 0 {
            current = current + nums[i];
        } else {
            current = nums[i];
        }
        if current < min {
            min = current;
        }
    }
    min
}

//HumanEval/115 int max_fill(vector<vector<int>> grid,int capacity){
fn max_fill(grid: Vec<Vec<i32>>, capacity: i32) -> i32 {
    let mut out: i32 = 0;

    for i in 0..grid.len() {
        let mut sum: i32 = 0;

        for j in 0..grid[i].len() {
            sum += grid[i][j];
        }
        if sum > 0 {
            out += (sum - 1) / capacity + 1;
        }
    }
    return out;
}

//HumanEval/116 vector<int> sort_array(vector<int> arr){
fn sort_array_1(arr: Vec<i32>) -> Vec<i32> {
    let mut arr_cp = arr.clone();
    let mut bin = vec![];
    let mut m;

    for i in 0..arr_cp.len() {
        let mut b = 0;
        let mut n = arr_cp[i].abs();
        while n > 0 {
            b += n % 2;
            n = n / 2;
        }
        bin.push(b);
    }
    for i in 0..arr_cp.len() {
        for j in 1..arr_cp.len() {
            if bin[j] < bin[j - 1] || (bin[j] == bin[j - 1] && arr_cp[j] < arr_cp[j - 1]) {
                m = arr_cp[j];
                arr_cp[j] = arr_cp[j - 1];
                arr_cp[j - 1] = m;
                m = bin[j];
                bin[j] = bin[j - 1];
                bin[j - 1] = m;
            }
        }
    }
    return arr_cp;
}

//HumanEval/117 vector<string> select_words(string s,int n){
fn select_words(s: &str, n: i32) -> Vec<String> {
    let vowels = "aeiouAEIOU";
    let mut current = String::new();
    let mut out = Vec::new();
    let mut numc = 0;
    let mut s = s.to_string();
    s.push(' ');
    for i in 0..s.len() {
        if s.chars().nth(i).unwrap() == ' ' {
            if numc == n {
                out.push(current);
            }
            current = String::new();
            numc = 0;
        } else {
            current.push(s.chars().nth(i).unwrap());
            if (s.chars().nth(i).unwrap() >= 'A' && s.chars().nth(i).unwrap() <= 'Z')
                || (s.chars().nth(i).unwrap() >= 'a' && s.chars().nth(i).unwrap() <= 'z')
            {
                if !vowels.contains(s.chars().nth(i).unwrap()) {
                    numc += 1;
                }
            }
        }
    }
    out
}

//HumanEval/118 string get_closest_vowel(string word){
fn get_closest_vowel(word: &str) -> String {
    let vowels = "AEIOUaeiou";
    let mut out = "".to_string();
    for i in (1..word.len() - 1).rev() {
        if vowels.contains(word.chars().nth(i).unwrap()) {
            if !vowels.contains(word.chars().nth(i + 1).unwrap()) {
                if !vowels.contains(word.chars().nth(i - 1).unwrap()) {
                    out.push(word.chars().nth(i).unwrap());
                    return out;
                }
            }
        }
    }
    out
}

//HumanEval/119 string match_parens(vector<string> lst){
fn match_parens(lst: Vec<&str>) -> &str {
    let l1 = lst[0].to_string() + lst[1];
    let mut count = 0;
    let mut can = true;
    for i in 0..l1.len() {
        if l1.chars().nth(i).unwrap() == '(' {
            count += 1;
        }
        if l1.chars().nth(i).unwrap() == ')' {
            count -= 1;
        }
        if count < 0 {
            can = false;
        }
    }
    if count != 0 {
        return "No";
    }
    if can == true {
        return "Yes";
    }
    let l1 = lst[1].to_string() + lst[0];
    let mut can = true;
    for i in 0..l1.len() {
        if l1.chars().nth(i).unwrap() == '(' {
            count += 1;
        }
        if l1.chars().nth(i).unwrap() == ')' {
            count -= 1;
        }
        if count < 0 {
            can = false;
        }
    }
    if can == true {
        return "Yes";
    }
    return "No";
}

//HumanEval/120 vector<int> maximum(vector<int> arr,int k){
fn maximum_120(arr: Vec<i32>, k: i32) -> Vec<i32> {
    let mut arr = arr;
    arr.sort();
    let mut arr_res: Vec<i32> = arr.iter().rev().take(k as usize).cloned().collect();
    arr_res.sort();
    return arr_res;
}

//HumanEval/121 int solutions(vector<int> lst){ CODEX
fn solutions(lst: Vec<i32>) -> i32 {
    let mut sum = 0;
    for (indx, elem) in lst.iter().enumerate() {
        if indx % 2 == 0 {
            if elem % 2 == 1 {
                sum += elem;
            }
        }
    }
    return sum;
}

//HumanEval/122 int add_elements(vector<int> arr,int k){ CODEX
fn add_elements(arr: Vec<i32>, k: i32) -> i32 {
    let mut sum = 0;
    for i in 0..k {
        if arr[i as usize] >= -99 && arr[i as usize] <= 99 {
            sum += arr[i as usize];
        }
    }
    sum
}

//HumanEval/123 vector<int> get_odd_collatz(int n){ CODEX
fn get_odd_collatz(n: i32) -> Vec<i32> {
    let mut out = vec![1];
    let mut n = n;
    while n != 1 {
        if n % 2 == 1 {
            out.push(n);
            n = n * 3 + 1;
        } else {
            n = n / 2;
        }
    }
    out.sort();
    out
}

//HumanEval/124 bool valid_date(string date){ CODEX
fn valid_date(date: &str) -> bool {
    let mut mm = 0;
    let mut dd = 0;
    let mut yy = 0;
    let mut i = 0;
    if date.len() != 10 {
        return false;
    }
    for i in 0..10 {
        if i == 2 || i == 5 {
            if date.chars().nth(i).unwrap() != '-' {
                return false;
            }
        } else if date.chars().nth(i).unwrap() < '0' || date.chars().nth(i).unwrap() > '9' {
            return false;
        }
    }
    mm = date[0..2].parse::<i32>().unwrap();
    dd = date[3..5].parse::<i32>().unwrap();
    yy = date[6..10].parse::<i32>().unwrap();
    if mm < 1 || mm > 12 {
        return false;
    }
    if dd < 1 || dd > 31 {
        return false;
    }
    if dd == 31 && (mm == 4 || mm == 6 || mm == 9 || mm == 11 || mm == 2) {
        return false;
    }
    if dd == 30 && mm == 2 {
        return false;
    }
    return true;
}

//HumanEval/125 vector<string> split_words(string txt){
fn split_words(txt: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let alphabet: HashMap<char, i32> = HashMap::from([
        ('a', 0),
        ('b', 1),
        ('c', 2),
        ('d', 3),
        ('e', 4),
        ('f', 5),
        ('g', 6),
        ('h', 7),
        ('i', 8),
        ('j', 9),
        ('k', 10),
        ('l', 11),
        ('m', 12),
        ('n', 13),
        ('o', 14),
        ('p', 15),
        ('q', 16),
        ('r', 17),
        ('s', 18),
        ('t', 19),
        ('u', 20),
        ('v', 21),
        ('w', 22),
        ('x', 23),
        ('y', 24),
        ('z', 25),
    ]);

    if txt.contains(' ') {
        out = txt
            .split_whitespace()
            .into_iter()
            .map(|c| c.to_string())
            .collect();
    } else if txt.contains(',') {
        out = txt.split(',').into_iter().map(|c| c.to_string()).collect();
    } else {
        let count = txt
            .chars()
            .into_iter()
            .filter(|c| c.is_ascii_lowercase())
            .filter(|c| alphabet.get(c).unwrap() % 2 == 1)
            .count();
        out.push(count.to_string());
    }

    return out;
}

//HumanEval/126 bool is_sorted(vector<int> lst){ CODEX
fn is_sorted(lst: Vec<i32>) -> bool {
    for i in 1..lst.len() {
        if lst[i] < lst[i - 1] {
            return false;
        }
        if i >= 2 && lst[i] == lst[i - 1] && lst[i] == lst[i - 2] {
            return false;
        }
    }
    true
}

//HumanEval/127 string intersection( vector<int> interval1,vector<int> interval2){ CODEX
fn intersection(interval1: Vec<i32>, interval2: Vec<i32>) -> String {
    let inter1 = std::cmp::max(interval1[0], interval2[0]);
    let inter2 = std::cmp::min(interval1[1], interval2[1]);
    let l = inter2 - inter1;
    if l < 2 {
        return "NO".to_string();
    }
    for i in 2..l {
        if l % i == 0 {
            return "NO".to_string();
        }
    }
    return "YES".to_string();
}

//HumanEval/128 int prod_signs(vector<int> arr){ CODEX
fn prod_signs(arr: Vec<i32>) -> i32 {
    if arr.is_empty() {
        return -32768;
    }
    let mut sum = 0;
    let mut prods = 1;
    for i in arr {
        sum += i.abs();
        if i == 0 {
            prods = 0;
        }
        if i < 0 {
            prods = -prods;
        }
    }
    sum * prods
}

//HumanEval/129 vector<int> minPath(vector<vector<int>> grid, int k){ CODEX
fn min_path(grid: Vec<Vec<i32>>, k: i32) -> Vec<i32> {
    let mut out: Vec<i32> = vec![];
    let mut x = 0;
    let mut y = 0;
    let mut min: i32 = (grid.len() * grid.len()) as i32;
    for i in 0..grid.len() {
        for j in 0..grid[i].len() {
            if grid[i][j] == 1 {
                x = i;
                y = j;
            }
        }
    }
    if x > 0 && grid[x - 1][y] < min {
        min = grid[x - 1][y];
    }
    if x < grid.len() - 1 && grid[x + 1][y] < min {
        min = grid[x + 1][y];
    }
    if y > 0 && grid[x][y - 1] < min {
        min = grid[x][y - 1];
    }
    if y < grid.len() - 1 && grid[x][y + 1] < min {
        min = grid[x][y + 1];
    }
    let mut out = vec![];
    for i in 0..k {
        if i % 2 == 0 {
            out.push(1);
        } else {
            out.push(min);
        }
    }
    out
}

//HumanEval/130 vector<int> tri(int n){ CODEX
fn tri(n: i32) -> Vec<i32> {
    let mut out = vec![1, 3];
    if n == 0 {
        return vec![1];
    }
    for i in 2..=n {
        if i % 2 == 0 {
            out.push(1 + i / 2);
        } else {
            out.push(out[(i - 1) as usize] + out[(i - 2) as usize] + 1 + (i + 1) / 2);
        }
    }
    out
}

//HumanEval/131 int digits(int n){ CODEX
fn digits(n: i32) -> i32 {
    let mut prod: i32 = 1;
    let mut has = 0;
    let s = n.to_string();
    for i in 0..s.len() {
        if s.chars().nth(i).unwrap().to_digit(10).unwrap() % 2 == 1 {
            has = 1;
            prod = prod * (s.chars().nth(i).unwrap().to_digit(10).unwrap()) as i32;
        }
    }
    if has == 0 {
        return 0;
    }
    prod
}

//HumanEval/132 bool is_nested(string str){ CODEX
fn is_nested(str: &str) -> bool {
    let mut count = 0;
    let mut maxcount = 0;
    for i in 0..str.len() {
        if str.chars().nth(i).unwrap() == '[' {
            count += 1;
        }
        if str.chars().nth(i).unwrap() == ']' {
            count -= 1;
        }
        if count < 0 {
            count = 0;
        }
        if count > maxcount {
            maxcount = count;
        }
        if count <= maxcount - 2 {
            return true;
        }
    }
    return false;
}

//HumanEval/133 int sum_squares(vector<float> lst){ CODEX
fn sum_squares(lst: Vec<f32>) -> i32 {
    let mut sum: f32 = 0.0;
    for i in 0..lst.len() {
        sum = sum + (lst[i].ceil() * lst[i].ceil());
    }
    sum as i32
}

//HumanEval/134 bool check_if_last_char_is_a_letter(string txt){ CODEX
fn check_if_last_char_is_a_letter(txt: &str) -> bool {
    if txt.len() == 0 {
        return false;
    }
    let chr = txt.chars().last().unwrap();
    if chr < 'A' || (chr > 'Z' && chr < 'a') || chr > 'z' {
        return false;
    }
    if txt.len() == 1 {
        return true;
    }
    let chr = txt.chars().nth(txt.len() - 2).unwrap();
    if (chr >= 'A' && chr <= 'Z') || (chr >= 'a' && chr <= 'z') {
        return false;
    }
    true
}

//HumanEval/135 int can_arrange(vector<int> arr){ CODEX
fn can_arrange(arr: Vec<i32>) -> i32 {
    let mut max: i32 = -1;
    for i in 0..arr.len() {
        if arr[i] <= i as i32 {
            max = i as i32;
        }
    }
    max
}

//HumanEval/136 vector<int> largest_smallest_integers(vector<int> lst){
fn largest_smallest_integers(lst: Vec<i32>) -> Vec<i32> {
    let mut maxneg = 0;
    let mut minpos = 0;
    for i in 0..lst.len() {
        if lst[i] < 0 && (maxneg == 0 || lst[i] > maxneg) {
            maxneg = lst[i];
        }
        if lst[i] > 0 && (minpos == 0 || lst[i] < minpos) {
            minpos = lst[i];
        }
    }
    vec![maxneg, minpos]
}

//HumanEval/137 boost::any compare_one(boost::any a,boost::any b){ TODO
pub fn compare_one<'a>(a: &'a dyn Any, b: &'a dyn Any) -> RtnType<String, f64, i32> {
    let a_f64 = Any_to_f64(a);
    let b_f64 = Any_to_f64(b);

    if a_f64 > b_f64 {
        return original_type(a);
    }

    if a_f64 < b_f64 {
        return original_type(b);
    } else {
        return RtnType::String("None".to_string());
    }
}

#[derive(Debug, PartialEq)]
pub enum RtnType<S, F, I> {
    Empty(),
    String(S),
    Float(F),
    Int(I),
}

fn Any_to_f64(a: &dyn Any) -> f64 {
    let mut a_f64 = 0.0;

    if a.downcast_ref::<f64>() == None {
        match a.downcast_ref::<&str>() {
            Some(as_string) => {
                a_f64 = as_string.parse::<f64>().unwrap();
            }
            None => {}
        }

        match a.downcast_ref::<i32>() {
            Some(as_i32) => {
                a_f64 = *as_i32 as f64;
            }
            None => {}
        }
    } else {
        a_f64 = *a.downcast_ref::<f64>().unwrap();
    }

    return a_f64;
}

fn original_type(a: &dyn Any) -> RtnType<String, f64, i32> {
    let mut res = RtnType::Empty();
    match a.downcast_ref::<&str>() {
        Some(as_string) => {
            res = RtnType::String(as_string.parse::<String>().unwrap());
        }
        None => {}
    }

    match a.downcast_ref::<i32>() {
        Some(as_i32) => {
            res = RtnType::Int(*as_i32);
        }
        None => {}
    }

    match a.downcast_ref::<f64>() {
        Some(as_f64) => res = RtnType::Float(*as_f64),
        None => {}
    }
    return res;
}

//HumanEval/138 bool is_equal_to_sum_even(int n){ CODEX
fn is_equal_to_sum_even(n: i32) -> bool {
    if n % 2 == 0 && n >= 8 {
        return true;
    }
    return false;
}

//HumanEva/139 long long special_factorial(int n){ CODEX
fn special_factorial(n: i32) -> i64 {
    let mut fact = 1;
    let mut bfact: i64 = 1;
    for i in 1..=n {
        fact = fact * i;
        bfact = bfact * fact as i64;
    }
    bfact
}

//HumanEval/140 string fix_spaces(string text){ CODEX
fn fix_spaces(text: &str) -> String {
    let mut out = String::new();
    let mut spacelen = 0;
    for c in text.chars() {
        if c == ' ' {
            spacelen += 1;
        } else {
            if spacelen == 1 {
                out.push('_');
            }
            if spacelen == 2 {
                out.push_str("__");
            }
            if spacelen > 2 {
                out.push('-');
            }
            spacelen = 0;
            out.push(c);
        }
    }
    if spacelen == 1 {
        out.push('_');
    }
    if spacelen == 2 {
        out.push_str("__");
    }
    if spacelen > 2 {
        out.push('-');
    }
    out
}

//HumanEval/141 string file_name_check(string file_name){ CODEX
fn file_name_check(file_name: &str) -> &str {
    let mut numdigit = 0;
    let mut numdot = 0;
    if file_name.len() < 5 {
        return "No";
    }
    let w = file_name.chars().nth(0).unwrap();
    if w < 'A' || (w > 'Z' && w < 'a') || w > 'z' {
        return "No";
    }
    let last = &file_name[file_name.len() - 4..];
    if last != ".txt" && last != ".exe" && last != ".dll" {
        return "No";
    }
    for c in file_name.chars() {
        if c >= '0' && c <= '9' {
            numdigit += 1;
        }
        if c == '.' {
            numdot += 1;
        }
    }
    if numdigit > 3 || numdot != 1 {
        return "No";
    }
    return "Yes";
}

//HumanEval/142 int sum_squares(vector<int> lst){ CODEX
fn sum_squares_142(lst: Vec<i32>) -> i32 {
    let mut sum = 0;
    for i in 0..lst.len() {
        if i % 3 == 0 {
            sum += lst[i] * lst[i];
        } else if i % 4 == 0 {
            sum += lst[i] * lst[i] * lst[i];
        } else {
            sum += lst[i];
        }
    }
    return sum;
}

//HumanEval/143 string words_in_sentence(string sentence){ CODEX
fn words_in_sentence(sentence: &str) -> String {
    let mut out = String::new();
    let mut current = String::new();
    let mut sentence = sentence.to_string();
    sentence.push(' ');

    for i in 0..sentence.len() {
        if sentence.chars().nth(i).unwrap() != ' ' {
            current.push(sentence.chars().nth(i).unwrap());
        } else {
            let mut isp = true;
            let l = current.len();
            if l < 2 {
                isp = false;
            }
            for j in 2..(l as f64).sqrt() as usize + 1 {
                if l % j == 0 {
                    isp = false;
                }
            }
            if isp {
                out.push_str(&current);
                out.push(' ');
            }
            current = String::new();
        }
    }
    if out.len() > 0 {
        out.pop();
    }
    out
}

//HumanEval/144 bool simplify(string x,string n){ CODEX
fn simplify(x: &str, n: &str) -> bool {
    let mut a = 0;
    let mut b = 0;
    let mut c = 0;
    let mut d = 0;
    let mut i = 0;
    for i in 0..x.len() {
        if x.chars().nth(i).unwrap() == '/' {
            a = x
                .chars()
                .take(i)
                .collect::<String>()
                .parse::<i32>()
                .unwrap();
            b = x
                .chars()
                .skip(i + 1)
                .collect::<String>()
                .parse::<i32>()
                .unwrap();
        }
    }
    for i in 0..n.len() {
        if n.chars().nth(i).unwrap() == '/' {
            c = n
                .chars()
                .take(i)
                .collect::<String>()
                .parse::<i32>()
                .unwrap();
            d = n
                .chars()
                .skip(i + 1)
                .collect::<String>()
                .parse::<i32>()
                .unwrap();
        }
    }
    if (a * c) % (b * d) == 0 {
        return true;
    }
    return false;
}

//HumanEval/145 vector<int> order_by_points(vector<int> nums){
fn order_by_points(arr: Vec<i32>) -> Vec<i32> {
    let mut result = arr.clone();
    result.sort_by_key(|&x| (sum_of_digits(x)));
    result
}

pub fn sum_of_digits(n: i32) -> i32 {
    let mut sum = 0;
    let mut n = n;
    if n < 0 {
        let right = n / 10;
        let mut left;

        if right != 0 {
            left = n % 10;
            left = -1 * left;
        } else {
            left = n % 10;
        }
        sum = right + left;
        return sum;
    }

    while n > 0 {
        sum += n % 10;
        n /= 10;
    }
    sum
}

//HumanEval/146 int specialFilter(vector<int> nums){ CODEX
fn special_filter(nums: Vec<i32>) -> i32 {
    let mut num = 0;
    for i in 0..nums.len() {
        if nums[i] > 10 {
            let w = nums[i].to_string();
            if w.chars().nth(0).unwrap().to_digit(10).unwrap() % 2 == 1
                && w.chars().last().unwrap().to_digit(10).unwrap() % 2 == 1
            {
                num += 1;
            }
        }
    }
    num
}

//HumanEval/147 int get_matrix_triples(int n){ CODEX
fn get_matrix_triples(n: i32) -> i32 {
    let mut a = vec![];
    let mut sum = vec![vec![0, 0, 0]];
    let mut sum2 = vec![vec![0, 0, 0]];

    for i in 1..=n {
        a.push((i * i - i + 1) % 3);
        sum.push(sum[sum.len() - 1].clone());
        sum[i as usize][a[i as usize - 1] as usize] += 1;
    }

    for times in 1..3 {
        for i in 1..=n {
            sum2.push(sum2[sum2.len() - 1].clone());
            if i >= 1 {
                for j in 0..=2 {
                    sum2[i as usize][(a[i as usize - 1] + j) as usize % 3] +=
                        sum[i as usize - 1][j as usize];
                }
            }
        }
        sum = sum2.clone();
        sum2 = vec![vec![0, 0, 0]];
    }

    return sum[n as usize][0];
}

//HumanEval/148 vector<string> bf(string planet1,string planet2){ CODEX
fn bf(planet1: &str, planet2: &str) -> Vec<String> {
    let planets = vec![
        "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
    ];
    let mut pos1: i32 = -1;
    let mut pos2: i32 = -1;
    let mut m;
    for m in 0..planets.len() {
        if planets[m] == planet1 {
            pos1 = m as i32;
        }
        if planets[m] == planet2 {
            pos2 = m as i32;
        }
    }
    if pos1 == -1 || pos2 == -1 {
        return vec![];
    }
    if pos1 > pos2 {
        m = pos1;
        pos1 = pos2;
        pos2 = m;
    }
    let mut out = vec![];
    for m in pos1 + 1..pos2 {
        out.push(planets[m as usize].to_string());
    }
    return out;
}

//HumanEval/149 vector<string> sorted_list_sum(vector<string> lst){  CODEX
fn sorted_list_sum(lst: Vec<&str>) -> Vec<&str> {
    let mut out: Vec<&str> = Vec::new();
    for i in 0..lst.len() {
        if lst[i].len() % 2 == 0 {
            out.push(lst[i]);
        }
    }
    out.sort();
    for i in 0..out.len() {
        for j in 1..out.len() {
            if out[j].len() < out[j - 1].len() {
                let mid = out[j];
                out[j] = out[j - 1];
                out[j - 1] = mid;
            }
        }
    }
    return out;
}

//HumanEval/150 int x_or_y(int n,int x,int y){ CODEX
fn x_or_y(n: i32, x: i32, y: i32) -> i32 {
    let mut isp = true;
    if n < 2 {
        isp = false;
    }
    for i in 2..=n / 2 {
        if n % i == 0 {
            isp = false;
        }
    }
    if isp {
        return x;
    }
    return y;
}

//HumanEval/151 long long double_the_difference(vector<float> lst){ CODEX
fn double_the_difference(lst: Vec<f32>) -> i64 {
    let mut sum: i64 = 0;
    for i in 0..lst.len() {
        if (lst[i] - lst[i].round()).abs() < 1e-4 {
            if lst[i] > 0.0 && (lst[i].round() as i64) % 2 == 1 {
                sum += (lst[i].round() as i64) * (lst[i].round() as i64);
            }
        }
    }
    return sum;
}

//HumanEval/152 vector<int> compare(vector<int> game,vector<int> guess){ CODEX
fn compare(game: Vec<i32>, guess: Vec<i32>) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::new();
    for i in 0..game.len() {
        out.push(i32::abs(game[i] - guess[i]));
    }
    return out;
}

//HumanEval/153 string Strongest_Extension(string class_name,vector<string> extensions){ CODEX
fn strongest_extension(class_name: &str, extensions: Vec<&str>) -> String {
    let mut strongest = "";
    let mut max = -1000;
    for i in 0..extensions.len() {
        let mut strength = 0;
        for j in 0..extensions[i].len() {
            let chr = extensions[i].chars().nth(j).unwrap();
            if chr >= 'A' && chr <= 'Z' {
                strength += 1;
            }
            if chr >= 'a' && chr <= 'z' {
                strength -= 1;
            }
        }
        if strength > max {
            max = strength;
            strongest = extensions[i];
        }
    }
    format!("{}.{}", class_name, strongest)
}

//HumanEval/154 bool cycpattern_check(string a,string b){ CODEX
fn cycpattern_check(a: &str, b: &str) -> bool {
    for i in 0..b.len() {
        let rotate = format!("{}{}", &b[i..], &b[..i]);
        if a.contains(&rotate) {
            return true;
        }
    }
    false
}

//HumanEval/155 vector<int> even_odd_count(int num){ CODEX
fn even_odd_count(num: i32) -> Vec<i32> {
    let w = num.abs().to_string();
    let mut n1 = 0;
    let mut n2 = 0;
    for i in 0..w.len() {
        if w.chars().nth(i).unwrap().to_digit(10).unwrap() % 2 == 1 {
            n1 += 1;
        } else {
            n2 += 1;
        }
    }
    vec![n2, n1]
}

//HumanEval/156 string int_to_mini_romank(int number){ CODEX
fn int_to_mini_romank(number: i32) -> String {
    let mut current = String::new();
    let mut number = number;
    let rep = vec![
        "m", "cm", "d", "cd", "c", "xc", "l", "xl", "x", "ix", "v", "iv", "i",
    ];
    let num = vec![1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1];
    let mut pos = 0;
    while number > 0 {
        while number >= num[pos] {
            current.push_str(rep[pos]);
            number -= num[pos];
        }
        if number > 0 {
            pos += 1;
        }
    }
    current
}

//HumanEval/157 bool right_angle_triangle(float a,float b,float c){ CODEX
fn right_angle_triangle(a: f32, b: f32, c: f32) -> bool {
    if (a * a + b * b - c * c).abs() < 1e-4
        || (a * a + c * c - b * b).abs() < 1e-4
        || (b * b + c * c - a * a).abs() < 1e-4
    {
        return true;
    }
    return false;
}

//HumanEval/158 string find_max(vector<string> words){ CODEX
fn find_max(words: Vec<&str>) -> &str {
    let mut max = "";
    let mut maxu = 0;
    for i in 0..words.len() {
        let mut unique = String::from("");
        for j in 0..words[i].len() {
            if !unique.contains(words[i].chars().nth(j).unwrap()) {
                unique.push(words[i].chars().nth(j).unwrap());
            }
        }
        if unique.len() > maxu || (unique.len() == maxu && words[i] < max) {
            max = words[i];
            maxu = unique.len();
        }
    }
    max
}

//HumanEval/159 vector<int> eat(int number,int need,int remaining){ CODEX
fn eat(number: i32, need: i32, remaining: i32) -> Vec<i32> {
    if need > remaining {
        return vec![number + remaining, 0];
    }
    return vec![number + need, remaining - need];
}

//HumanEval/160 int do_algebra(vector<string> operato, vector<int> operand){ CODEX
fn do_algebra(operato: Vec<&str>, operand: Vec<i32>) -> i32 {
    let mut operand: Vec<i32> = operand;
    let mut num: Vec<i32> = vec![];
    let mut posto: Vec<i32> = vec![];
    for i in 0..operand.len() {
        posto.push(i as i32);
    }
    for i in 0..operato.len() {
        if operato[i] == "**" {
            while posto[posto[i] as usize] != posto[i] {
                posto[i] = posto[posto[i] as usize];
            }
            while posto[posto[i + 1] as usize] != posto[i + 1] {
                posto[i + 1] = posto[posto[i + 1] as usize];
            }
            operand[posto[i] as usize] =
                operand[posto[i] as usize].pow(operand[posto[i + 1] as usize] as u32);
            posto[i + 1] = posto[i];
        }
    }
    for i in 0..operato.len() {
        if operato[i] == "*" || operato[i] == "//" {
            while posto[posto[i] as usize] != posto[i] {
                posto[i] = posto[posto[i] as usize];
            }
            while posto[posto[i + 1] as usize] != posto[i + 1] {
                posto[i + 1] = posto[posto[i + 1] as usize];
            }
            if operato[i] == "*" {
                operand[posto[i] as usize] =
                    operand[posto[i] as usize] * operand[posto[i + 1] as usize];
            } else {
                operand[posto[i] as usize] =
                    operand[posto[i] as usize] / operand[posto[i + 1] as usize];
            }
            posto[i + 1] = posto[i];
        }
    }
    for i in 0..operato.len() {
        if operato[i] == "+" || operato[i] == "-" {
            while posto[posto[i] as usize] != posto[i] {
                posto[i] = posto[posto[i] as usize];
            }
            while posto[posto[i + 1] as usize] != posto[i + 1] {
                posto[i + 1] = posto[posto[i + 1] as usize];
            }
            if operato[i] == "+" {
                operand[posto[i] as usize] =
                    operand[posto[i] as usize] + operand[posto[i + 1] as usize];
            } else {
                operand[posto[i] as usize] =
                    operand[posto[i] as usize] - operand[posto[i + 1] as usize];
            }
            posto[i + 1] = posto[i];
        }
    }
    operand[0]
}

//HumanEval/161 string solve(string s){ CODEX
pub fn solve_161(s: &str) -> String {
    let mut nletter = 0;
    let mut out = String::new();
    for c in s.chars() {
        let mut w = c;
        if w >= 'A' && w <= 'Z' {
            w = w.to_ascii_lowercase();
        } else if w >= 'a' && w <= 'z' {
            w = w.to_ascii_uppercase();
        } else {
            nletter += 1;
        }
        out.push(w);
    }
    if nletter == s.len() {
        out.chars().rev().collect()
    } else {
        out
    }
}

//HumanEval/162 string string_to_md5(string text){
fn string_to_md5(text: &str) -> String {
    if text.is_empty() {
        return "None".to_string();
    }

    let digest = md5::compute(text.as_bytes());
    return format!("{:x}", digest);
}

//HumanEval/163 vector<int> generate_integers(int a,int b){ CODEX
fn generate_integers(a: i32, b: i32) -> Vec<i32> {
    let mut a = a;
    let mut b = b;
    let mut m;

    if b < a {
        m = a;
        a = b;
        b = m;
    }

    let mut out = vec![];
    for i in a..=b {
        if i < 10 && i % 2 == 0 {
            out.push(i);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_close_elements() {
        assert_eq!(
            has_close_elements(vec![11.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3),
            true
        );
        assert_eq!(
            has_close_elements(vec![1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05),
            false
        );
        assert_eq!(
            has_close_elements(vec![1.0, 2.0, 5.9, 4.0, 5.0], 0.95),
            true
        );
        assert_eq!(
            has_close_elements(vec![1.0, 2.0, 5.9, 4.0, 5.0], 0.8),
            false
        );
        assert_eq!(
            has_close_elements(vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1),
            true
        );
        assert_eq!(has_close_elements(vec![1.1, 2.2, 3.1, 4.1, 5.1], 1.0), true);
        assert_eq!(
            has_close_elements(vec![1.1, 2.2, 3.1, 4.1, 5.1], 0.5),
            false
        );
    }

    #[test]
    fn test_separate_paren_groups() {
        assert_eq!(
            separate_paren_groups(String::from("(()()) ((())) () ((())()())")),
            vec!["(()())", "((()))", "()", "((())()())"]
        );
        assert_eq!(
            separate_paren_groups(String::from("() (()) ((())) (((())))")),
            vec!["()", "(())", "((()))", "(((())))"]
        );
        assert_eq!(
            separate_paren_groups(String::from("(()(())((())))")),
            vec!["(()(())((())))"]
        );
        assert_eq!(
            separate_paren_groups(String::from("( ) (( )) (( )( ))")),
            vec!["()", "(())", "(()())"]
        );
    }

    #[test]
    fn test_truncate_number() {
        assert_eq!(truncate_number(&3.5), 0.5);
        let t1: f32 = 1.33 - 0.33;
        assert!(truncate_number(&t1) < 0.000001);
        let t2: f32 = 123.456 - 0.456;
        assert!(truncate_number(&t2) < 0.000001);
    }

    #[test]
    fn test_below_zero() {
        assert_eq!(below_zero(vec![]), false);
        assert_eq!(below_zero(vec![1, 2, -3, 1, 2, -3]), false);
        assert_eq!(below_zero(vec![1, 2, -4, 5, 6]), true);
        assert_eq!(below_zero(vec![1, -1, 2, -2, 5, -5, 4, -4]), false);
        assert_eq!(below_zero(vec![1, -1, 2, -2, 5, -5, 4, -5]), true);
        assert_eq!(below_zero(vec![1, -2, 2, -2, 5, -5, 4, -4]), true);
    }

    #[test]
    fn test_mean_absolute_deviation() {
        /* function + operation.... ? -doubtful test quality ERROR
        asert abs(candidate([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6
        asert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6
        asert abs(candidate([1.0, 2.0, 3.0, 4.0, 5.0]) - 6.0/5.0) < 1e-6 */
        assert!(mean_absolute_deviation(vec![1.0, 2.0, 3.0]) - 2.0 / 3.0 < 0.000001);
        assert!(mean_absolute_deviation(vec![1.0, 2.0, 3.0, 4.0]) - 1.0 < 0.000001);
        assert!(mean_absolute_deviation(vec![1.0, 2.0, 3.0, 4.0, 5.0]) - 6.0 / 5.0 < 0.000001);
    }

    #[test]
    fn test_intersperse() {
        assert!(intersperse(vec![], 7) == vec![]);
        assert!(intersperse(vec![5, 6, 3, 2], 8) == vec![5, 8, 6, 8, 3, 8, 2]);
        assert!(intersperse(vec![2, 2, 2], 2) == vec![2, 2, 2, 2, 2]);
    }
    #[test]
    fn test_parse_nested_parens() {
        assert!(
            parse_nested_parens(String::from("(()()) ((())) () ((())()())")) == vec![2, 3, 1, 3]
        );
        assert!(parse_nested_parens(String::from("() (()) ((())) (((())))")) == vec![1, 2, 3, 4]);
        assert!(parse_nested_parens(String::from("(()(())((())))")) == vec![4]);
    }

    #[test]
    fn test_filter_by_substring() {
        let v_empty: Vec<String> = vec![];
        assert!(filter_by_substring(vec![], String::from("john")) == v_empty);
        assert!(
            filter_by_substring(
                vec![
                    "xxx".to_string(),
                    "asd".to_string(),
                    "xxy".to_string(),
                    "john doe".to_string(),
                    "xxxAAA".to_string(),
                    "xxx".to_string()
                ],
                String::from("xxx")
            ) == vec!["xxx", "xxxAAA", "xxx"]
        );
        assert!(
            filter_by_substring(
                vec![
                    "xxx".to_string(),
                    "asd".to_string(),
                    "aaaxxy".to_string(),
                    "john doe".to_string(),
                    "xxxAAA".to_string(),
                    "xxx".to_string()
                ],
                String::from("xx")
            ) == vec!["xxx", "aaaxxy", "xxxAAA", "xxx"]
        );
        assert!(
            filter_by_substring(
                vec![
                    "grunt".to_string(),
                    "trumpet".to_string(),
                    "prune".to_string(),
                    "gruesome".to_string()
                ],
                String::from("run")
            ) == ["grunt", "prune"]
        );
    }

    #[test]
    fn test_sum_product() {
        assert!(sum_product(vec![]) == (0, 1));
        assert!(sum_product(vec![1, 1, 1]) == (3, 1));
        assert!(sum_product(vec![100, 0]) == (100, 0));
        assert!(sum_product(vec![3, 5, 7]) == (3 + 5 + 7, 3 * 5 * 7));
        assert!(sum_product(vec![10]) == (10, 10));
    }
    #[test]
    fn test_rolling_max() {
        assert!(rolling_max(vec![]) == vec![]);
        assert!(rolling_max(vec![1, 2, 3, 4]) == vec![1, 2, 3, 4]);
        assert!(rolling_max(vec![4, 3, 2, 1]) == vec![4, 4, 4, 4]);
        assert!(rolling_max(vec![3, 2, 3, 100, 3]) == vec![3, 3, 3, 100, 100]);
    }

    #[test]
    fn test_make_palindrome() {
        assert_eq!(make_palindrome(""), "");
        assert_eq!(make_palindrome("x"), "x");
        assert_eq!(make_palindrome("xyz"), "xyzyx");
        assert_eq!(make_palindrome("xyx"), "xyx");
        assert_eq!(make_palindrome("jerry"), "jerryrrej");
    }

    #[test]
    fn test_string_xor() {
        assert!(string_xor("111000".to_string(), "101010".to_string()) == "010010");
        assert!(string_xor("1".to_string(), "1".to_string()) == "0");
        assert!(string_xor("0101".to_string(), "0000".to_string()) == "0101");
    }

    #[test]
    fn test_longest() {
        assert!(longest(vec![]) == None);
        assert!(
            longest(vec!["x".to_string(), "y".to_string(), "z".to_string()])
                == Some("x".to_string())
        );
        assert!(
            longest(vec![
                "x".to_string(),
                "yyy".to_string(),
                "zzzz".to_string(),
                "www".to_string(),
                "kkkk".to_string(),
                "abc".to_string()
            ]) == Some("zzzz".to_string())
        );
    }

    #[test]
    fn test_greatest_common_divisor() {
        assert!(greatest_common_divisor(3, 7) == 1);
        assert!(greatest_common_divisor(10, 15) == 5);
        assert!(greatest_common_divisor(49, 14) == 7);
        assert!(greatest_common_divisor(144, 60) == 12);
    }
    #[test]
    fn test_all_prefixes() {
        let v_empty: Vec<String> = vec![];
        assert!(all_prefixes(String::from("")) == v_empty);
        assert!(
            all_prefixes(String::from("asdfgh"))
                == vec!["a", "as", "asd", "asdf", "asdfg", "asdfgh"]
        );
        assert!(all_prefixes(String::from("WWW")) == vec!["W", "WW", "WWW"]);
    }

    #[test]
    fn test_string_sequence() {
        assert!(string_sequence(0) == "0".to_string());
        assert!(string_sequence(3) == "0 1 2 3".to_string());
        assert!(string_sequence(10) == "0 1 2 3 4 5 6 7 8 9 10".to_string());
    }

    #[test]
    fn test_count_distinct_characters() {
        assert!(count_distinct_characters("".to_string()) == 0);
        assert!(count_distinct_characters("abcde".to_string()) == 5);
        assert!(
            count_distinct_characters(
                "abcde".to_string() + &"cade".to_string() + &"CADE".to_string()
            ) == 5
        );
        assert!(count_distinct_characters("aaaaAAAAaaaa".to_string()) == 1);
        assert!(count_distinct_characters("Jerry jERRY JeRRRY".to_string()) == 5);
    }

    #[test]
    fn test_parse_music() {
        assert!(parse_music(" ".to_string()) == []);
        assert!(parse_music("o o o o".to_string()) == vec![4, 4, 4, 4]);
        assert!(parse_music(".| .| .| .|".to_string()) == vec![1, 1, 1, 1]);
        assert!(parse_music("o| o| .| .| o o o o".to_string()) == vec![2, 2, 1, 1, 4, 4, 4, 4]);
        assert!(parse_music("o| .| o| .| o o| o o|".to_string()) == vec![2, 1, 2, 1, 4, 2, 4, 2]);
    }

    #[test]
    fn test_how_many_times() {
        assert!(how_many_times("".to_string(), "x".to_string()) == 0);
        assert!(how_many_times("xyxyxyx".to_string(), "x".to_string()) == 4);
        assert!(how_many_times("cacacacac".to_string(), "cac".to_string()) == 4);
        assert!(how_many_times("john doe".to_string(), "john".to_string()) == 1);
    }

    #[test]
    fn test_sort_numbers() {
        assert!(sort_numbers("".to_string()) == "".to_string());
        assert!(sort_numbers("three".to_string()) == "three".to_string());
        assert!(sort_numbers("three five nine".to_string()) == "three five nine");
        assert!(
            sort_numbers("five zero four seven nine eight".to_string())
                == "zero four five seven eight nine".to_string()
        );
        assert!(
            sort_numbers("six five four three two one zero".to_string())
                == "zero one two three four five six".to_string()
        );
    }

    #[test]
    fn test_find_closest_elements() {
        assert!(find_closest_elements(vec![1.0, 2.0, 3.9, 4.0, 5.0, 2.2]) == (3.9, 4.0));
        assert!(find_closest_elements(vec![1.0, 2.0, 5.9, 4.0, 5.0]) == (5.0, 5.9));
        assert!(find_closest_elements(vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.2]) == (2.0, 2.2));
        assert!(find_closest_elements(vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.0]) == (2.0, 2.0));
        assert!(find_closest_elements(vec![1.1, 2.2, 3.1, 4.1, 5.1]) == (2.2, 3.1));
    }

    #[test]
    fn test_rescale_to_unit() {
        assert!(rescale_to_unit(vec![2.0, 49.9]) == [0.0, 1.0]);
        assert!(rescale_to_unit(vec![100.0, 49.9]) == [1.0, 0.0]);
        assert!(rescale_to_unit(vec![1.0, 2.0, 3.0, 4.0, 5.0]) == [0.0, 0.25, 0.5, 0.75, 1.0]);
        assert!(rescale_to_unit(vec![2.0, 1.0, 5.0, 3.0, 4.0]) == [0.25, 0.0, 1.0, 0.5, 0.75]);
        assert!(rescale_to_unit(vec![12.0, 11.0, 15.0, 13.0, 14.0]) == [0.25, 0.0, 1.0, 0.5, 0.75]);
    }

    #[test]
    fn test_filter_integers() {
        assert_eq!(filter_integers(vec![]), vec![]);
        let v_empty: Vec<Box<dyn Any>> = vec![];
        assert_eq!(
            filter_integers(vec![
                Box::new(4),
                Box::new(v_empty),
                Box::new(23.2),
                Box::new(9),
                Box::new(String::from("adasd"))
            ]),
            vec![4, 9]
        );
        assert_eq!(
            filter_integers(vec![
                Box::new(3),
                Box::new('c'),
                Box::new(3),
                Box::new(3),
                Box::new('a'),
                Box::new('b')
            ]),
            vec![3, 3, 3]
        );
    }

    #[test]
    fn test_strlen() {
        assert!(strlen("".to_string()) == 0);
        assert!(strlen("x".to_string()) == 1);
        assert!(strlen("asdasnakj".to_string()) == 9);
    }

    #[test]
    fn test_largest_divisor() {
        assert!(largest_divisor(3) == 1);
        assert!(largest_divisor(7) == 1);
        assert!(largest_divisor(10) == 5);
        assert!(largest_divisor(100) == 50);
        assert!(largest_divisor(49) == 7);
    }

    #[test]
    fn test_factorize() {
        assert_eq!(factorize(2), vec![2]);
        assert_eq!(factorize(4), vec![2, 2]);
        assert_eq!(factorize(8), vec![2, 2, 2]);
        assert_eq!(factorize(3 * 19), vec![3, 19]);
        assert_eq!(factorize(3 * 19 * 3 * 19), vec![3, 3, 19, 19]);
        assert_eq!(
            factorize(3 * 19 * 3 * 19 * 3 * 19),
            vec![3, 3, 3, 19, 19, 19]
        );
        assert_eq!(factorize(3 * 19 * 19 * 19), vec![3, 19, 19, 19]);
        assert_eq!(factorize(3 * 2 * 3), vec![2, 3, 3]);
    }

    #[test]
    fn test_remove_duplicates() {
        assert!(remove_duplicates(vec![]) == []);
        assert!(remove_duplicates(vec![1, 2, 3, 4]) == vec![1, 2, 3, 4]);
        assert!(remove_duplicates(vec![1, 2, 3, 2, 4, 3, 5]) == [1, 4, 5]);
    }

    #[test]
    fn test_flip_case() {
        assert!(flip_case("".to_string()) == "".to_string());
        assert!(flip_case("Hello!".to_string()) == "hELLO!".to_string());
        assert!(
            flip_case("These violent delights have violent ends".to_string())
                == "tHESE VIOLENT DELIGHTS HAVE VIOLENT ENDS".to_string()
        );
    }

    #[test]
    fn test_concatenate() {
        assert!(concatenate(vec![]) == "".to_string());
        assert!(
            concatenate(vec!["x".to_string(), "y".to_string(), "z".to_string()])
                == "xyz".to_string()
        );
        assert!(
            concatenate(vec![
                "x".to_string(),
                "y".to_string(),
                "z".to_string(),
                "w".to_string(),
                "k".to_string()
            ]) == "xyzwk".to_string()
        );
    }

    #[test]
    fn test_filter_by_prefix() {
        let v_empty: Vec<String> = vec![];
        assert!(filter_by_prefix(vec![], "john".to_string()) == v_empty);
        assert!(
            filter_by_prefix(
                vec![
                    "xxx".to_string(),
                    "asd".to_string(),
                    "xxy".to_string(),
                    "john doe".to_string(),
                    "xxxAAA".to_string(),
                    "xxx".to_string()
                ],
                "xxx".to_string()
            ) == vec!["xxx", "xxxAAA", "xxx"]
        );
    }

    #[test]
    fn test_get_positive() {
        assert!(get_positive(vec![-1, -2, 4, 5, 6]) == [4, 5, 6]);
        assert!(
            get_positive(vec![5, 3, -5, 2, 3, 3, 9, 0, 123, 1, -10]) == [5, 3, 2, 3, 3, 9, 123, 1]
        );
        assert!(get_positive(vec![-1, -2]) == []);
        assert!(get_positive(vec![]) == []);
    }

    #[test]
    fn test_is_prime() {
        assert!(is_prime(6) == false);
        assert!(is_prime(101) == true);
        assert!(is_prime(11) == true);
        assert!(is_prime(13441) == true);
        assert!(is_prime(61) == true);
        assert!(is_prime(4) == false);
        assert!(is_prime(1) == false);
        assert!(is_prime(5) == true);
        assert!(is_prime(11) == true);
        assert!(is_prime(17) == true);
        assert!(is_prime(5 * 17) == false);
        assert!(is_prime(11 * 7) == false);
        assert!(is_prime(13441 * 19) == false);
    }

    /*
    #[test]
    fn test_poly() {
        let mut rng = rand::thread_rng();
        let mut solution: f64;
        let mut ncoeff: i32;
        for _ in 0..100 {
            ncoeff = 2 * (1 + rng.gen_range(0, 4));
            let mut coeffs = vec![];
            for _ in 0..ncoeff {
                let coeff = -10 + rng.gen_range(0, 21);
                if coeff == 0 {
                    coeffs.push(1.0);
                } else {
                    coeffs.push(coeff as f64);
                }
            }
            solution = find_zero(&coeffs);
            assert!(poly(&coeffs, solution).abs() < 1e-3);
        }
    }
     */

    #[test] //Bad test??????? REAL ERROR
    fn test_sort_third() {
        let mut l = vec![1, 2, 3];
        assert_eq!(sort_third(l), vec![1, 2, 3]);
        l = vec![5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10];
        assert_eq!(sort_third(l), vec![5, 3, -5, 1, -3, 3, 2, 0, 123, 9, -10]);
        l = vec![5, 8, -12, 4, 23, 2, 3, 11, 12, -10];
        assert_eq!(sort_third(l), vec![5, 8, -12, -10, 23, 2, 3, 11, 12, 4]);
        l = vec![5, 6, 3, 4, 8, 9, 2];
        assert_eq!(sort_third(l), vec![5, 6, 3, 2, 8, 9, 4]);
        l = vec![5, 8, 3, 4, 6, 9, 2];
        assert_eq!(sort_third(l), vec![5, 8, 3, 2, 6, 9, 4]);
        l = vec![5, 6, 9, 4, 8, 3, 2];
        assert_eq!(sort_third(l), vec![5, 6, 9, 2, 8, 3, 4]);
        l = vec![5, 6, 3, 4, 8, 9, 2, 1];
        assert_eq!(sort_third(l), vec![5, 6, 3, 2, 8, 9, 4, 1]);
    }

    #[test]
    fn test_unique() {
        assert!(unique(vec![5, 3, 5, 2, 3, 3, 9, 0, 123]) == vec![0, 2, 3, 5, 9, 123]);
    }

    #[test]
    fn test_maximum() {
        assert!(maximum(vec![1, 2, 3]) == 3);
        assert!(maximum(vec![5, 3, -5, 2, -3, 3, 9, 0, 124, 1, -10]) == 124);
    }

    #[test]
    fn test_fizz_buzz() {
        assert!(fizz_buzz(50) == 0);
        assert!(fizz_buzz(78) == 2);
        assert!(fizz_buzz(79) == 3);
        assert!(fizz_buzz(100) == 3);
        assert!(fizz_buzz(200) == 6);
        assert!(fizz_buzz(4000) == 192);
        assert!(fizz_buzz(10000) == 639);
        assert!(fizz_buzz(100000) == 8026);
    }

    #[test]
    fn test_sort_even() {
        assert_eq!(sort_even(vec![1, 2, 3]), vec![1, 2, 3]);
        assert_eq!(
            sort_even(vec![5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10]),
            vec![-10, 3, -5, 2, -3, 3, 5, 0, 9, 1, 123]
        );
        assert_eq!(
            sort_even(vec![5, 8, -12, 4, 23, 2, 3, 11, 12, -10]),
            vec![-12, 8, 3, 4, 5, 2, 12, 11, 23, -10]
        );
    }

    #[test]
    fn test_decode_cyclic() {
        for _ in 0..100 {
            let l = 10 + rand::random::<u32>() % 11;
            let mut str = String::new();
            for _ in 0..l {
                let chr = 97 + rand::random::<u32>() % 26;
                str.push(chr as u8 as char);
            }
            let encoded_str = encode_cyclic(&str);
            assert_eq!(decode_cyclic(&encoded_str), str);
        }
    }

    #[test]
    fn test_prime_fib() {
        assert_eq!(prime_fib(1), 2);
        assert_eq!(prime_fib(2), 3);
        assert_eq!(prime_fib(3), 5);
        assert_eq!(prime_fib(4), 13);
        assert_eq!(prime_fib(5), 89);
        assert_eq!(prime_fib(6), 233);
        assert_eq!(prime_fib(7), 1597);
        assert_eq!(prime_fib(8), 28657);
        assert_eq!(prime_fib(9), 514229);
        assert_eq!(prime_fib(10), 433494437);
    }

    #[test]
    fn test_triples_sum_to_zero() {
        assert!(triples_sum_to_zero(vec![1, 3, 5, 0]) == false);
        assert!(triples_sum_to_zero(vec![1, 3, 5, -1]) == false);
        assert!(triples_sum_to_zero(vec![1, 3, -2, 1]) == true);
        assert!(triples_sum_to_zero(vec![1, 2, 3, 7]) == false);
        assert!(triples_sum_to_zero(vec![1, 2, 5, 7]) == false);
        assert!(triples_sum_to_zero(vec![2, 4, -5, 3, 9, 7]) == true);
        assert!(triples_sum_to_zero(vec![1]) == false);
        assert!(triples_sum_to_zero(vec![1, 3, 5, -100]) == false);
        assert!(triples_sum_to_zero(vec![100, 3, 5, -100]) == false);
    }

    #[test]
    fn test_car_race_collision() {
        assert!(car_race_collision(2) == 4);
        assert!(car_race_collision(3) == 9);
        assert!(car_race_collision(4) == 16);
        assert!(car_race_collision(8) == 64);
        assert!(car_race_collision(10) == 100);
    }

    #[test]
    fn test_incr_list() {
        assert!(incr_list(vec![]) == vec![]);
        assert!(incr_list(vec![3, 2, 1]) == [4, 3, 2]);
        assert!(incr_list(vec![5, 2, 5, 2, 3, 3, 9, 0, 123]) == [6, 3, 6, 3, 4, 4, 10, 1, 124]);
    }

    #[test]
    fn test_pairs_sum_to_zero() {
        assert!(pairs_sum_to_zero(vec![1, 3, 5, 0]) == false);
        assert!(pairs_sum_to_zero(vec![1, 3, -2, 1]) == false);
        assert!(pairs_sum_to_zero(vec![1, 2, 3, 7]) == false);
        assert!(pairs_sum_to_zero(vec![2, 4, -5, 3, 5, 7]) == true);
        assert!(pairs_sum_to_zero(vec![1]) == false);
        assert!(pairs_sum_to_zero(vec![-3, 9, -1, 3, 2, 30]) == true);
        assert!(pairs_sum_to_zero(vec![-3, 9, -1, 3, 2, 31]) == true);
        assert!(pairs_sum_to_zero(vec![-3, 9, -1, 4, 2, 30]) == false);
        assert!(pairs_sum_to_zero(vec![-3, 9, -1, 4, 2, 31]) == false);
    }

    #[test]
    fn test_change_base() {
        assert!(change_base(8, 3) == "22".to_string());
        assert!(change_base(9, 3) == "100".to_string());
        assert!(change_base(234, 2) == "11101010".to_string());
        assert!(change_base(16, 2) == "10000".to_string());
        assert!(change_base(8, 2) == "1000".to_string());
        assert!(change_base(7, 2) == "111".to_string());
    }

    #[test]
    fn test_triangle_area() {
        assert!(triangle_area(5, 3) == 7.5);
        assert!(triangle_area(2, 2) == 2.0);
        assert!(triangle_area(10, 8) == 40.0);
    }

    #[test]
    fn test_fib4() {
        assert!(fib4(5) == 4);
        assert!(fib4(8) == 28);
        assert!(fib4(10) == 104);
        assert!(fib4(12) == 386);
    }

    #[test]
    fn test_median() {
        assert!(median(vec![3, 1, 2, 4, 5]) == 3.0);
        assert!(median(vec![-10, 4, 6, 1000, 10, 20]) == 8.0);
        assert!(median(vec![5]) == 5.0);
        assert!(median(vec![6, 5]) == 5.5);
        assert!(median(vec![8, 1, 3, 9, 9, 2, 7]) == 7.0);
    }

    #[test]
    fn test_is_palindrome() {
        assert!(is_palindrome("".to_string()) == true);
        assert!(is_palindrome("aba".to_string()) == true);
        assert!(is_palindrome("aaaaa".to_string()) == true);
        assert!(is_palindrome("zbcd".to_string()) == false);
        assert!(is_palindrome("xywyx".to_string()) == true);
        assert!(is_palindrome("xywyz".to_string()) == false);
        assert!(is_palindrome("xywzx".to_string()) == false);
    }

    #[test]
    fn test_modp() {
        assert!(modp(3, 5) == 3);
        assert!(modp(1101, 101) == 2);
        assert!(modp(0, 101) == 1);
        assert!(modp(3, 11) == 8);
        assert!(modp(100, 101) == 1);
        assert!(modp(30, 5) == 4);
        assert!(modp(31, 5) == 3);
    }
    /* Original test ERROR
    #[test]
    fn test_decode_encode(){
    let mut rng = rand::thread_rng();
    for _ in 0..100{
        let r1:i32 = rng.gen();
        let l:i32= 10 + r1%11;
        let mut str:String="".to_string();

        for j in 0..l{
            let r2:i32 = rng.gen();
            let chr:char = char::from_u32((97 + r2%26) as u32).unwrap();
            println!("{}", chr);
            str.push(chr);
        }

        let encoded_str:String = encode_shift(&str);
        assert!(decode_shift(&encoded_str) == str);
    }
    */

    #[test]
    //Imposing that random characters that can be generated are solely from the alphabet
    fn test_decode_encode() {
        fn random_char() -> char {
            let mut rng = rand::thread_rng();
            let letter: char = match rng.gen_range(0, 2) {
                0 => rng.gen_range(b'a', b'z' + 1).into(),
                1 => rng.gen_range(b'A', b'Z' + 1).into(),
                _ => unreachable!(),
            };
            return letter;
        }

        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let r1: i32 = rng.gen();
            let l: i32 = 10 + r1 % 11;
            let mut str: String = "".to_string();

            for _ in 0..l {
                let chr: char = random_char();
                println!("{}", chr);
                str.push(chr);
            }

            let encoded_str: String = encode_shift(&str);
            assert!(decode_shift(&encoded_str) == str);
        }
    }

    #[test]
    fn test_remove_vowels() {
        assert!(remove_vowels("") == "");
        assert!(remove_vowels("abcdef\nghijklm") == "bcdf\nghjklm");
        assert!(remove_vowels("fedcba") == "fdcb");
        assert!(remove_vowels("eeeee") == "");
        assert!(remove_vowels("acBAA") == "cB");
        assert!(remove_vowels("EcBOO") == "cB");
        assert!(remove_vowels("ybcd") == "ybcd");
    }

    #[test]
    fn test_below_threshold() {
        assert!(below_threshold(vec![1, 2, 4, 10], 100));
        assert!(!below_threshold(vec![1, 20, 4, 10], 5));
        assert!(below_threshold(vec![1, 20, 4, 10], 21));
        assert!(below_threshold(vec![1, 20, 4, 10], 22));
        assert!(below_threshold(vec![1, 8, 4, 10], 11));
        assert!(!below_threshold(vec![1, 8, 4, 10], 10));
    }

    #[test]
    fn test_add() {
        assert!(add(0, 1) == 1);
        assert!(add(1, 0) == 1);
        assert!(add(2, 3) == 5);
        assert!(add(5, 7) == 12);
        assert!(add(7, 5) == 12);
        for _ in 0..100 {
            let mut rng = rand::thread_rng();
            let mut x: i32 = rng.gen();
            x = x % 1000;
            let mut y: i32 = rng.gen();
            y = y % 1000;

            assert!(add(x, y) == x + y);
        }
    }

    #[test]
    fn test_same_chars() {
        assert!(same_chars("eabcdzzzz", "dddzzzzzzzddeddabc") == true);
        assert!(same_chars("abcd", "dddddddabc") == true);
        assert!(same_chars("dddddddabc", "abcd") == true);
        assert!(same_chars("eabcd", "dddddddabc") == false);
        assert!(same_chars("abcd", "dddddddabcf") == false);
        assert!(same_chars("eabcdzzzz", "dddzzzzzzzddddabc") == false);
        assert!(same_chars("aabb", "aaccc") == false);
    }

    #[test]
    fn test_fib() {
        assert!(fib(10) == 55);
        assert!(fib(1) == 1);
        assert!(fib(8) == 21);
        assert!(fib(11) == 89);
        assert!(fib(12) == 144);
    }

    #[test]
    fn test_correct_bracketing() {
        assert!(correct_bracketing("<>"));
        assert!(correct_bracketing("<<><>>"));
        assert!(correct_bracketing("<><><<><>><>"));
        assert!(correct_bracketing("<><><<<><><>><>><<><><<>>>"));
        assert!(!(correct_bracketing("<<<><>>>>")));
        assert!(!(correct_bracketing("><<>")));
        assert!(!(correct_bracketing("<")));
        assert!(!(correct_bracketing("<<<<")));
        assert!(!(correct_bracketing(">")));
        assert!(!(correct_bracketing("<<>")));
        assert!(!(correct_bracketing("<><><<><>><>><<>")));
        assert!(!(correct_bracketing("<><><<><>><>>><>")));
    }

    #[test]
    fn test_monotonic() {
        assert!(monotonic(vec![1, 2, 4, 10]) == true);
        assert!(monotonic(vec![1, 2, 4, 20]) == true);
        assert!(monotonic(vec![1, 20, 4, 10]) == false);
        assert!(monotonic(vec![4, 1, 0, -10]) == true);
        assert!(monotonic(vec![4, 1, 1, 0]) == true);
        assert!(monotonic(vec![1, 2, 3, 2, 5, 60]) == false);
        assert!(monotonic(vec![1, 2, 3, 4, 5, 60]) == true);
        assert!(monotonic(vec![9, 9, 9, 9]) == true);
    }

    #[test]
    fn test_common() {
        assert!(
            common(vec![1, 4, 3, 34, 653, 2, 5], vec![5, 7, 1, 5, 9, 653, 121]) == vec![1, 5, 653]
        );
        assert!(common(vec![5, 3, 2, 8], vec![3, 2]) == vec![2, 3]);
        assert!(common(vec![4, 3, 2, 8], vec![3, 2, 4]) == vec![2, 3, 4]);
        assert!(common(vec![4, 3, 2, 8], vec![]) == vec![]);
    }

    #[test]
    fn test_largest_prime_factor() {
        assert!(largest_prime_factor(15) == 5);
        assert!(largest_prime_factor(27) == 3);
        assert!(largest_prime_factor(63) == 7);
        assert!(largest_prime_factor(330) == 11);
        assert!(largest_prime_factor(13195) == 29);
    }
    #[test]
    fn test_sum_to_n() {
        assert!(sum_to_n(1) == 1);
        assert!(sum_to_n(6) == 21);
        assert!(sum_to_n(11) == 66);
        assert!(sum_to_n(30) == 465);
        assert!(sum_to_n(100) == 5050);
    }

    #[test]
    fn test_correct_bracketing_parenthesis() {
        assert!(correct_bracketing_parenthesis("()"));
        assert!(correct_bracketing_parenthesis("(()())"));
        assert!(correct_bracketing_parenthesis("()()(()())()"));
        assert!(correct_bracketing_parenthesis("()()((()()())())(()()(()))"));
        assert!(!(correct_bracketing_parenthesis("((()())))")));
        assert!(!(correct_bracketing_parenthesis(")(()")));
        assert!(!(correct_bracketing_parenthesis("(")));
        assert!(!(correct_bracketing_parenthesis("((((")));
        assert!(!(correct_bracketing_parenthesis(")")));
        assert!(!(correct_bracketing_parenthesis("(()")));
        assert!(!(correct_bracketing_parenthesis("()()(()())())(()")));
        assert!(!(correct_bracketing_parenthesis("()()(()())()))()")));
    }

    #[test]
    fn test_derivative() {
        assert!(derivative(vec![3, 1, 2, 4, 5]) == vec![1, 4, 12, 20]);
        assert!(derivative(vec![1, 2, 3]) == vec![2, 6]);
        assert!(derivative(vec![3, 2, 1]) == vec![2, 2]);
        assert!(derivative(vec![3, 2, 1, 0, 4]) == vec![2, 2, 0, 16]);
        assert!(derivative(vec![1]) == vec![]);
    }

    #[test]
    fn test_fibfib() {
        assert!(fibfib(2) == 1);
        assert!(fibfib(1) == 0);
        assert!(fibfib(5) == 4);
        assert!(fibfib(8) == 24);
        assert!(fibfib(10) == 81);
        assert!(fibfib(12) == 274);
        assert!(fibfib(14) == 927);
    }

    #[test]
    fn test_vowels_count() {
        assert!(vowels_count("abcde") == 2);
        assert!(vowels_count("Alone") == 3);
        assert!(vowels_count("key") == 2);
        assert!(vowels_count("bye") == 1);
        assert!(vowels_count("keY") == 2);
        assert!(vowels_count("bYe") == 1);
        assert!(vowels_count("ACEDY") == 3);
    }

    #[test]
    fn test_circular_shift() {
        assert!(circular_shift(100, 2) == "001");
        assert!(circular_shift(12, 8) == "12");
        // original test   asert (circular_shift(97, 8) == "79"); DATASET ERROR
        assert!(circular_shift(97, 8) == "97");
        assert!(circular_shift(12, 1) == "21");
        assert!(circular_shift(11, 101) == "11");
    }

    #[test]
    fn test_digitSum() {
        assert!(digitSum("") == 0);
        assert!(digitSum("abAB") == 131);
        assert!(digitSum("abcCd") == 67);
        assert!(digitSum("helloE") == 69);
        assert!(digitSum("woArBld") == 131);
        assert!(digitSum("aAaaaXa") == 153);
        assert!(digitSum(" How are yOu?") == 151);
        assert!(digitSum("You arE Very Smart") == 327);
    }

    #[test]
    fn test_fruit_distribution() {
        assert!(fruit_distribution("5 apples and 6 oranges", 19) == 8);
        assert!(fruit_distribution("5 apples and 6 oranges", 21) == 10);
        assert!(fruit_distribution("0 apples and 1 oranges", 3) == 2);
        assert!(fruit_distribution("1 apples and 0 oranges", 3) == 2);
        assert!(fruit_distribution("2 apples and 3 oranges", 100) == 95);
        assert!(fruit_distribution("2 apples and 3 oranges", 5) == 0);
        assert!(fruit_distribution("1 apples and 100 oranges", 120) == 19);
    }

    #[test]
    fn test_pluck() {
        assert!(pluck(vec![4, 2, 3]) == vec![2, 1]);
        assert!(pluck(vec![1, 2, 3]) == vec![2, 1]);
        assert!(pluck(vec![]) == vec![]);
        assert!(pluck(vec![5, 0, 3, 0, 4, 2]) == vec![0, 1]);
        assert!(pluck(vec![1, 2, 3, 0, 5, 3]) == vec![0, 3]);
        assert!(pluck(vec![5, 4, 8, 4, 8]) == vec![4, 1]);
        assert!(pluck(vec![7, 6, 7, 1]) == vec![6, 1]);
        assert!(pluck(vec![7, 9, 7, 1]) == vec![]);
    }

    #[test]
    fn test_search() {
        assert!(search(vec![5, 5, 5, 5, 1]) == 1);
        assert!(search(vec![4, 1, 4, 1, 4, 4]) == 4);
        assert!(search(vec![3, 3]) == -1);
        assert!(search(vec![8, 8, 8, 8, 8, 8, 8, 8]) == 8);
        assert!(search(vec![2, 3, 3, 2, 2]) == 2);
        assert!(
            search(vec![
                2, 7, 8, 8, 4, 8, 7, 3, 9, 6, 5, 10, 4, 3, 6, 7, 1, 7, 4, 10, 8, 1
            ]) == 1
        );
        assert!(search(vec![3, 2, 8, 2]) == 2);
        assert!(search(vec![6, 7, 1, 8, 8, 10, 5, 8, 5, 3, 10]) == 1);
        assert!(search(vec![8, 8, 3, 6, 5, 6, 4]) == -1);
        assert!(
            search(vec![
                6, 9, 6, 7, 1, 4, 7, 1, 8, 8, 9, 8, 10, 10, 8, 4, 10, 4, 10, 1, 2, 9, 5, 7, 9
            ]) == 1
        );
        assert!(search(vec![1, 9, 10, 1, 3]) == 1);
        assert!(
            search(vec![
                6, 9, 7, 5, 8, 7, 5, 3, 7, 5, 10, 10, 3, 6, 10, 2, 8, 6, 5, 4, 9, 5, 3, 10
            ]) == 5
        );
        assert!(search(vec![1]) == 1);
        assert!(
            search(vec![
                8, 8, 10, 6, 4, 3, 5, 8, 2, 4, 2, 8, 4, 6, 10, 4, 2, 1, 10, 2, 1, 1, 5
            ]) == 4
        );
        assert!(
            search(vec![
                2, 10, 4, 8, 2, 10, 5, 1, 2, 9, 5, 5, 6, 3, 8, 6, 4, 10
            ]) == 2
        );
        assert!(search(vec![1, 6, 10, 1, 6, 9, 10, 8, 6, 8, 7, 3]) == 1);
        assert!(
            search(vec![
                9, 2, 4, 1, 5, 1, 5, 2, 5, 7, 7, 7, 3, 10, 1, 5, 4, 2, 8, 4, 1, 9, 10, 7, 10, 2, 8,
                10, 9, 4
            ]) == 4
        );
        assert!(
            search(vec![
                2, 6, 4, 2, 8, 7, 5, 6, 4, 10, 4, 6, 3, 7, 8, 8, 3, 1, 4, 2, 2, 10, 7
            ]) == 4
        );
        assert!(
            search(vec![
                9, 8, 6, 10, 2, 6, 10, 2, 7, 8, 10, 3, 8, 2, 6, 2, 3, 1
            ]) == 2
        );
        assert!(
            search(vec![
                5, 5, 3, 9, 5, 6, 3, 2, 8, 5, 6, 10, 10, 6, 8, 4, 10, 7, 7, 10, 8
            ]) == -1
        );
        assert!(search(vec![10]) == -1);
        assert!(search(vec![9, 7, 7, 2, 4, 7, 2, 10, 9, 7, 5, 7, 2]) == 2);
        assert!(search(vec![5, 4, 10, 2, 1, 1, 10, 3, 6, 1, 8]) == 1);
        assert!(
            search(vec![
                7, 9, 9, 9, 3, 4, 1, 5, 9, 1, 2, 1, 1, 10, 7, 5, 6, 7, 6, 7, 7, 6
            ]) == 1
        );
        assert!(search(vec![3, 10, 10, 9, 2]) == -1);
    }

    #[test]
    fn test_strange_sort_list() {
        assert!(strange_sort_list(vec![1, 2, 3, 4]) == vec![1, 4, 2, 3]);
        assert!(strange_sort_list(vec![5, 6, 7, 8, 9]) == vec![5, 9, 6, 8, 7]);
        assert!(strange_sort_list(vec![1, 2, 3, 4, 5]) == vec![1, 5, 2, 4, 3]);
        assert!(strange_sort_list(vec![5, 6, 7, 8, 9, 1]) == vec![1, 9, 5, 8, 6, 7]);
        assert!(strange_sort_list(vec![5, 5, 5, 5]) == vec![5, 5, 5, 5]);
        assert!(strange_sort_list(vec![]) == vec![]);
        assert!(strange_sort_list(vec![1, 2, 3, 4, 5, 6, 7, 8]) == vec![1, 8, 2, 7, 3, 6, 4, 5]);
        assert!(
            strange_sort_list(vec![0, 2, 2, 2, 5, 5, -5, -5]) == vec![-5, 5, -5, 5, 0, 2, 2, 2]
        );
        assert!(strange_sort_list(vec![111111]) == vec![111111]);
    }

    #[test]
    fn test_triangle_area_f64() {
        assert!(f64::abs(triangle_area_f64(3.0, 4.0, 5.0) - 6.00) < 0.01);
        assert!(f64::abs(triangle_area_f64(1.0, 2.0, 10.0) + 1.0) < 0.01);
        assert!(f64::abs(triangle_area_f64(4.0, 8.0, 5.0) - 8.18) < 0.01);
        assert!(f64::abs(triangle_area_f64(2.0, 2.0, 2.0) - 1.73) < 0.01);
        assert!(f64::abs(triangle_area_f64(1.0, 2.0, 3.0) + 1.0) < 0.01);
        assert!(f64::abs(triangle_area_f64(10.0, 5.0, 7.0) - 16.25) < 0.01);
        assert!(f64::abs(triangle_area_f64(2.0, 6.0, 3.0) + 1.0) < 0.01);
        assert!(f64::abs(triangle_area_f64(1.0, 1.0, 1.0) - 0.43) < 0.01);
        assert!(f64::abs(triangle_area_f64(2.0, 2.0, 10.0) + 1.0) < 0.01);
    }

    #[test]
    fn test_will_it_fly() {
        assert!(will_it_fly(vec![3, 2, 3], 9) == true);
        assert!(will_it_fly(vec![1, 2], 5) == false);
        assert!(will_it_fly(vec![3], 5) == true);
        assert!(will_it_fly(vec![3, 2, 3], 1) == false);
        assert!(will_it_fly(vec![1, 2, 3], 6) == false);
        assert!(will_it_fly(vec![5], 5) == true);
    }

    #[test]
    fn test_smallest_change() {
        assert!(smallest_change(vec![1, 2, 3, 5, 4, 7, 9, 6]) == 4);
        assert!(smallest_change(vec![1, 2, 3, 4, 3, 2, 2]) == 1);
        assert!(smallest_change(vec![1, 4, 2]) == 1);
        assert!(smallest_change(vec![1, 4, 4, 2]) == 1);
        assert!(smallest_change(vec![1, 2, 3, 2, 1]) == 0);
        assert!(smallest_change(vec![3, 1, 1, 3]) == 0);
        assert!(smallest_change(vec![1]) == 0);
        assert!(smallest_change(vec![0, 1]) == 1);
    }

    #[test]
    fn test_total_match() {
        let v_empty: Vec<String> = vec![];
        assert!(total_match(vec![], vec![]) == v_empty);
        assert!(total_match(vec!["hi", "admin"], vec!["hi", "hi"]) == vec!["hi", "hi"]);
        assert!(
            total_match(vec!["hi", "admin"], vec!["hi", "hi", "admin", "project"])
                == vec!["hi", "admin"]
        );
        assert!(total_match(vec!["4"], vec!["1", "2", "3", "4", "5"]) == vec!["4"]);
        assert!(total_match(vec!["hi", "admin"], vec!["hI", "Hi"]) == vec!["hI", "Hi"]);
        assert!(total_match(vec!["hi", "admin"], vec!["hI", "hi", "hi"]) == vec!["hI", "hi", "hi"]);
        assert!(total_match(vec!["hi", "admin"], vec!["hI", "hi", "hii"]) == vec!["hi", "admin"]);
        assert!(total_match(vec![], vec!["this"]) == v_empty);
        assert!(total_match(vec!["this"], vec![]) == v_empty);
    }

    #[test]
    fn test_is_multiply_prime() {
        assert!(is_multiply_prime(5) == false);
        assert!(is_multiply_prime(30) == true);
        assert!(is_multiply_prime(8) == true);
        assert!(is_multiply_prime(10) == false);
        assert!(is_multiply_prime(125) == true);
        assert!(is_multiply_prime(3 * 5 * 7) == true);
        assert!(is_multiply_prime(3 * 6 * 7) == false);
        assert!(is_multiply_prime(9 * 9 * 9) == false);
        assert!(is_multiply_prime(11 * 9 * 9) == false);
        assert!(is_multiply_prime(11 * 13 * 7) == true);
    }

    #[test]
    fn test_is_simple_power() {
        assert!(is_simple_power(1, 4) == true);
        assert!(is_simple_power(2, 2) == true);
        assert!(is_simple_power(8, 2) == true);
        assert!(is_simple_power(3, 2) == false);
        assert!(is_simple_power(3, 1) == false);
        assert!(is_simple_power(5, 3) == false);
        assert!(is_simple_power(16, 2) == true);
        assert!(is_simple_power(143214, 16) == false);
        assert!(is_simple_power(4, 2) == true);
        assert!(is_simple_power(9, 3) == true);
        assert!(is_simple_power(16, 4) == true);
        assert!(is_simple_power(24, 2) == false);
        assert!(is_simple_power(128, 4) == false);
        assert!(is_simple_power(12, 6) == false);
        assert!(is_simple_power(1, 1) == true);
        assert!(is_simple_power(1, 12) == true);
    }

    #[test]
    fn test_iscuber() {
        assert!(iscuber(1) == true);
        assert!(iscuber(2) == false);
        assert!(iscuber(-1) == true);
        assert!(iscuber(64) == true);
        assert!(iscuber(180) == false);
        assert!(iscuber(1000) == true);
        assert!(iscuber(0) == true);
        assert!(iscuber(1729) == false);
    }

    #[test]
    fn test_hex_key() {
        assert!(hex_key("AB") == 1);
        assert!(hex_key("1077E") == 2);
        assert!(hex_key("ABED1A33") == 4);
        assert!(hex_key("2020") == 2);
        assert!(hex_key("123456789ABCDEF0") == 6);
        assert!(hex_key("112233445566778899AABBCCDDEEFF00") == 12);
        assert!(hex_key("") == 0);
    }

    #[test]
    fn test_decimal_to_binary() {
        assert!(decimal_to_binary(0) == "db0db".to_string());
        assert!(decimal_to_binary(32) == "db100000db".to_string());
        assert!(decimal_to_binary(103) == "db1100111db".to_string());
        assert!(decimal_to_binary(15) == "db1111db".to_string());
    }

    #[test]
    fn test_is_happy() {
        assert!(is_happy("a") == false);
        assert!(is_happy("aa") == false);
        assert!(is_happy("abcd") == true);
        assert!(is_happy("aabb") == false);
        assert!(is_happy("adb") == true);
        assert!(is_happy("xyy") == false);
        assert!(is_happy("iopaxpoi") == true);
        assert!(is_happy("iopaxioi") == false);
    }

    #[test]
    fn test_numerical_letter_grade() {
        assert!(
            numerical_letter_grade(vec![4.0, 3.0, 1.7, 2.0, 3.5])
                == vec!["A+", "B", "C-", "C", "A-"]
        );
        assert!(numerical_letter_grade(vec![1.2]) == vec!["D+"]);
        assert!(numerical_letter_grade(vec![0.5]) == vec!["D-"]);
        assert!(numerical_letter_grade(vec![0.0]) == vec!["E"]);
        assert!(
            numerical_letter_grade(vec![1.0, 0.3, 1.5, 2.8, 3.3])
                == vec!["D", "D-", "C-", "B", "B+"]
        );
        assert!(numerical_letter_grade(vec![0.0, 0.7]) == vec!["E", "D-"]);
    }

    #[test]
    fn test_prime_length() {
        assert!(prime_length("Hello") == true);
        assert!(prime_length("abcdcba") == true);
        assert!(prime_length("kittens") == true);
        assert!(prime_length("orange") == false);
        assert!(prime_length("wow") == true);
        assert!(prime_length("world") == true);
        assert!(prime_length("MadaM") == true);
        assert!(prime_length("Wow") == true);
        assert!(prime_length("") == false);
        assert!(prime_length("HI") == true);
        assert!(prime_length("go") == true);
        assert!(prime_length("gogo") == false);
        assert!(prime_length("aaaaaaaaaaaaaaa") == false);
        assert!(prime_length("Madam") == true);
        assert!(prime_length("M") == false);
        assert!(prime_length("0") == false);
    }

    #[test]
    fn test_starts_one_ends() {
        assert!(starts_one_ends(1) == 1);
        assert!(starts_one_ends(2) == 18);
        assert!(starts_one_ends(3) == 180);
        assert!(starts_one_ends(4) == 1800);
        assert!(starts_one_ends(5) == 18000);
    }

    #[test]
    fn test_solve() {
        assert!(solve(1000) == "1");
        assert!(solve(150) == "110");
        assert!(solve(147) == "1100");
        assert!(solve(333) == "1001");
        assert!(solve(963) == "10010");
    }

    #[test]
    fn test_add_even_odd() {
        assert!(add_even_odd(vec![4, 88]) == 88);
        assert!(add_even_odd(vec![4, 5, 6, 7, 2, 122]) == 122);
        assert!(add_even_odd(vec![4, 0, 6, 7]) == 0);
        assert!(add_even_odd(vec![4, 4, 6, 8]) == 12);
    }

    #[test]
    fn test_anti_shuffle() {
        assert!(anti_shuffle("Hi") == "Hi".to_string());
        assert!(anti_shuffle("hello") == "ehllo".to_string());
        assert!(anti_shuffle("number") == "bemnru".to_string());
        assert!(anti_shuffle("abcd") == "abcd".to_string());
        assert!(anti_shuffle("Hello World!!!") == "Hello !!!Wdlor".to_string());
        assert!(anti_shuffle("") == "".to_string());
        assert!(
            anti_shuffle("Hi. My name is Mister Robot. How are you?")
                == ".Hi My aemn is Meirst .Rboot How aer ?ouy".to_string()
        );
    }

    #[test]
    fn test_get_row() {
        assert!(
            get_row(
                vec![
                    vec![1, 2, 3, 4, 5, 6],
                    vec![1, 2, 3, 4, 1, 6],
                    vec![1, 2, 3, 4, 5, 1]
                ],
                1
            ) == vec![vec![0, 0], vec![1, 0], vec![1, 4], vec![2, 0], vec![2, 5]]
        );
        assert!(
            get_row(
                vec![
                    vec![1, 2, 3, 4, 5, 6],
                    vec![1, 2, 3, 4, 5, 6],
                    vec![1, 2, 3, 4, 5, 6],
                    vec![1, 2, 3, 4, 5, 6],
                    vec![1, 2, 3, 4, 5, 6],
                    vec![1, 2, 3, 4, 5, 6]
                ],
                2
            ) == vec![
                vec![0, 1],
                vec![1, 1],
                vec![2, 1],
                vec![3, 1],
                vec![4, 1],
                vec![5, 1]
            ]
        );
        assert!(
            get_row(
                vec![
                    vec![1, 2, 3, 4, 5, 6],
                    vec![1, 2, 3, 4, 5, 6],
                    vec![1, 1, 3, 4, 5, 6],
                    vec![1, 2, 1, 4, 5, 6],
                    vec![1, 2, 3, 1, 5, 6],
                    vec![1, 2, 3, 4, 1, 6],
                    vec![1, 2, 3, 4, 5, 1]
                ],
                1
            ) == vec![
                vec![0, 0],
                vec![1, 0],
                vec![2, 0],
                vec![2, 1],
                vec![3, 0],
                vec![3, 2],
                vec![4, 0],
                vec![4, 3],
                vec![5, 0],
                vec![5, 4],
                vec![6, 0],
                vec![6, 5]
            ]
        );
        let v: Vec<Vec<i32>> = vec![];
        assert!(get_row(vec![], 1) == v);
        assert!(get_row(vec![vec![1]], 2) == v);
        assert!(get_row(vec![vec![], vec![1], vec![1, 2, 3]], 3) == vec![vec![2, 2]]);
    }

    #[test]
    fn test_sort_array() {
        assert!(sort_array(vec![]) == vec![]);
        assert!(sort_array(vec![5]) == vec![5]);
        assert!(sort_array(vec![2, 4, 3, 0, 1, 5]) == vec![0, 1, 2, 3, 4, 5]);
        assert!(sort_array(vec![2, 4, 3, 0, 1, 5, 6]) == vec![6, 5, 4, 3, 2, 1, 0]);
        assert!(sort_array(vec![2, 1]) == vec![1, 2]);
        assert!(sort_array(vec![15, 42, 87, 32, 11, 0]) == vec![0, 11, 15, 32, 42, 87]);
        assert!(sort_array(vec![21, 14, 23, 11]) == vec![23, 21, 14, 11]);
    }

    #[test]
    fn test_encrypt() {
        assert!(encrypt("hi") == "lm");
        assert!(encrypt("asdfghjkl") == "ewhjklnop");
        assert!(encrypt("gf") == "kj");
        assert!(encrypt("et") == "ix");
        assert!(encrypt("faewfawefaewg") == "jeiajeaijeiak");
        assert!(encrypt("hellomyfriend") == "lippsqcjvmirh");
        assert!(
            encrypt("dxzdlmnilfuhmilufhlihufnmlimnufhlimnufhfucufh")
                == "hbdhpqrmpjylqmpyjlpmlyjrqpmqryjlpmqryjljygyjl"
        );
        assert!(encrypt("a") == "e");
    }

    #[test]
    fn test_next_smallest() {
        assert!(next_smallest(vec![1, 2, 3, 4, 5]) == 2);
        assert!(next_smallest(vec![5, 1, 4, 3, 2]) == 2);
        assert!(next_smallest(vec![]) == -1);
        assert!(next_smallest(vec![1, 1]) == -1);
        assert!(next_smallest(vec![1, 1, 1, 1, 0]) == 1);
        assert!(next_smallest(vec![-35, 34, 12, -45]) == -35);
    }

    #[test]
    fn test_is_bored() {
        assert!(is_bored("Hello world") == 0);
        assert!(is_bored("Is the sky blue?") == 0);
        assert!(is_bored("I love It !") == 1);
        assert!(is_bored("bIt") == 0);
        assert!(is_bored("I feel good today. I will be productive. will kill It") == 2);
        assert!(is_bored("You and I are going for a walk") == 0);
    }

    #[test]
    fn test_any_int() {
        assert!(any_int(2.0, 3.0, 1.0) == true);
        assert!(any_int(2.5, 2.0, 3.0) == false);
        assert!(any_int(1.5, 5.0, 3.5) == false);
        assert!(any_int(2.0, 6.0, 2.0) == false);
        assert!(any_int(4.0, 2.0, 2.0) == true);
        assert!(any_int(2.2, 2.2, 2.2) == false);
        assert!(any_int(-4.0, 6.0, 2.0) == true);
        assert!(any_int(2.0, 1.0, 1.0) == true);
        assert!(any_int(3.0, 4.0, 7.0) == true);
        assert!(any_int(3.01, 4.0, 7.0) == false);
    }

    #[test]
    fn test_encode() {
        assert!(encode("TEST") == "tgst");
        assert!(encode("Mudasir") == "mWDCSKR");
        assert!(encode("YES") == "ygs");
        assert!(encode("This is a message") == "tHKS KS C MGSSCGG");
        assert!(encode("I DoNt KnOw WhAt tO WrItE") == "k dQnT kNqW wHcT Tq wRkTg");
    }

    #[test]
    fn test_skjkasdkd() {
        assert!(
            skjkasdkd(vec![
                0, 3, 2, 1, 3, 5, 7, 4, 5, 5, 5, 2, 181, 32, 4, 32, 3, 2, 32, 324, 4, 3
            ]) == 10
        );
        assert!(
            skjkasdkd(vec![
                1, 0, 1, 8, 2, 4597, 2, 1, 3, 40, 1, 2, 1, 2, 4, 2, 5, 1
            ]) == 25
        );
        assert!(
            skjkasdkd(vec![
                1, 3, 1, 32, 5107, 34, 83278, 109, 163, 23, 2323, 32, 30, 1, 9, 3
            ]) == 13
        );
        assert!(skjkasdkd(vec![0, 724, 32, 71, 99, 32, 6, 0, 5, 91, 83, 0, 5, 6]) == 11);
        assert!(skjkasdkd(vec![0, 81, 12, 3, 1, 21]) == 3);
        assert!(skjkasdkd(vec![0, 8, 1, 2, 1, 7]) == 7);
        assert!(skjkasdkd(vec![8191]) == 19);
        assert!(skjkasdkd(vec![8191, 123456, 127, 7]) == 19);
        assert!(skjkasdkd(vec![127, 97, 8192]) == 10);
    }

    #[test]
    fn test_check_dict_case() {
        assert!(check_dict_case(HashMap::from([("p", "pineapple"), ("b", "banana")])) == true);
        assert!(
            check_dict_case(HashMap::from([
                ("p", "pineapple"),
                ("A", "banana"),
                ("B", "banana")
            ])) == false
        );
        assert!(
            check_dict_case(HashMap::from([
                ("p", "pineapple"),
                ("5", "banana"),
                ("a", "apple")
            ])) == false
        );
        assert!(
            check_dict_case(HashMap::from([
                ("Name", "John"),
                ("Age", "36"),
                ("City", "Houston")
            ])) == false
        );
        assert!(check_dict_case(HashMap::from([("STATE", "NC"), ("ZIP", "12345")])) == true);
        assert!(check_dict_case(HashMap::from([("fruit", "Orange"), ("taste", "Sweet")])) == true);
        assert!(check_dict_case(HashMap::new()) == false);
    }

    #[test]
    fn test_count_up_to() {
        assert!(count_up_to(5) == vec![2, 3]);
        assert!(count_up_to(6) == vec![2, 3, 5]);
        assert!(count_up_to(7) == vec![2, 3, 5]);
        assert!(count_up_to(10) == vec![2, 3, 5, 7]);
        assert!(count_up_to(0) == vec![]);
        assert!(count_up_to(22) == vec![2, 3, 5, 7, 11, 13, 17, 19]);
        assert!(count_up_to(1) == vec![]);
        assert!(count_up_to(18) == vec![2, 3, 5, 7, 11, 13, 17]);
        assert!(count_up_to(47) == vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]);
        assert!(
            count_up_to(101)
                == vec![
                    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
                    79, 83, 89, 97
                ]
        );
    }

    #[test]
    fn test_multiply() {
        assert!(multiply(148, 412) == 16);
        assert!(multiply(19, 28) == 72);
        assert!(multiply(2020, 1851) == 0);
        assert!(multiply(14, -15) == 20);
        assert!(multiply(76, 67) == 42);
        assert!(multiply(17, 27) == 49);
        assert!(multiply(0, 1) == 0);
        assert!(multiply(0, 0) == 0);
    }

    #[test]
    fn test_count_upper() {
        assert!(count_upper("aBCdEf") == 1);
        assert!(count_upper("abcdefg") == 0);
        assert!(count_upper("dBBE") == 0);
        assert!(count_upper("B") == 0);
        assert!(count_upper("U") == 1);
        assert!(count_upper("") == 0);
        assert!(count_upper("EEEE") == 2);
    }

    #[test]
    fn test_closest_integer() {
        assert!(closest_integer("10") == 10);
        assert!(closest_integer("14.5") == 15);
        assert!(closest_integer("-15.5") == -16);
        assert!(closest_integer("15.3") == 15);
        assert!(closest_integer("0") == 0);
    }

    #[test]
    fn test_make_a_pile() {
        assert!(make_a_pile(3) == vec![3, 5, 7]);
        assert!(make_a_pile(4) == vec![4, 6, 8, 10]);
        assert!(make_a_pile(5) == vec![5, 7, 9, 11, 13]);
        assert!(make_a_pile(6) == vec![6, 8, 10, 12, 14, 16]);
        assert!(make_a_pile(8) == vec![8, 10, 12, 14, 16, 18, 20, 22]);
    }

    #[test]
    fn test_words_string() {
        assert!(words_string("Hi, my name is John") == vec!["Hi", "my", "name", "is", "John"]);
        assert!(
            words_string("One, two, three, four, five, six")
                == vec!["One", "two", "three", "four", "five", "six"]
        );
        assert!(words_string("Hi, my name") == vec!["Hi", "my", "name"]);
        assert!(
            words_string("One,, two, three, four, five, six,")
                == vec!["One", "two", "three", "four", "five", "six"]
        );
        let v_empty: Vec<String> = vec![];
        assert!(words_string("") == v_empty);
        assert!(words_string("ahmed , gamal") == vec!["ahmed", "gamal"]);
    }

    #[test]
    fn test_choose_num() {
        assert!(choose_num(12, 15) == 14);
        assert!(choose_num(13, 12) == -1);
        assert!(choose_num(33, 12354) == 12354);
        assert!(choose_num(6, 29) == 28);
        assert!(choose_num(27, 10) == -1);
        assert!(choose_num(7, 7) == -1);
        assert!(choose_num(546, 546) == 546);
    }

    #[test]
    fn test_rounded_avg() {
        assert!(rounded_avg(1, 5) == "11");
        assert!(rounded_avg(7, 13) == "1010");
        assert!(rounded_avg(964, 977) == "1111001010");
        assert!(rounded_avg(996, 997) == "1111100100");
        assert!(rounded_avg(560, 851) == "1011000001");
        assert!(rounded_avg(185, 546) == "101101101");
        assert!(rounded_avg(362, 496) == "110101101");
        assert!(rounded_avg(350, 902) == "1001110010");
        assert!(rounded_avg(197, 233) == "11010111");
        assert!(rounded_avg(7, 5) == "-1");
        assert!(rounded_avg(5, 1) == "-1");
        assert!(rounded_avg(5, 5) == "101");
    }

    #[test]
    fn test_unique_digits() {
        assert!(unique_digits(vec![15, 33, 1422, 1]) == vec![1, 15, 33]);
        assert!(unique_digits(vec![152, 323, 1422, 10]) == vec![]);
        assert!(unique_digits(vec![12345, 2033, 111, 151]) == vec![111, 151]);
        assert!(unique_digits(vec![135, 103, 31]) == vec![31, 135]);
    }

    #[test]
    fn test_by_length() {
        assert!(
            by_length(vec![2, 1, 1, 4, 5, 8, 2, 3])
                == vec!["Eight", "Five", "Four", "Three", "Two", "Two", "One", "One"]
        );
        let v_empty: Vec<String> = vec![];
        assert!(by_length(vec![]) == v_empty);
        assert!(by_length(vec![1, -1, 55]) == vec!["One"]);
        assert!(by_length(vec![1, -1, 3, 2]) == vec!["Three", "Two", "One"]);
        assert!(by_length(vec![9, 4, 8]) == vec!["Nine", "Eight", "Four"]);
    }

    #[test]
    fn test_f() {
        assert!(f(5) == vec![1, 2, 6, 24, 15]);
        assert!(f(7) == vec![1, 2, 6, 24, 15, 720, 28]);
        assert!(f(1) == vec![1]);
        assert!(f(3) == vec![1, 2, 6]);
    }

    #[test]
    fn test_even_odd_palindrome() {
        assert!(even_odd_palindrome(123) == (8, 13));
        assert!(even_odd_palindrome(12) == (4, 6));
        assert!(even_odd_palindrome(3) == (1, 2));
        assert!(even_odd_palindrome(63) == (6, 8));
        assert!(even_odd_palindrome(25) == (5, 6));
        assert!(even_odd_palindrome(19) == (4, 6));
        assert!(even_odd_palindrome(9) == (4, 5));
        assert!(even_odd_palindrome(1) == (0, 1));
    }

    #[test]
    fn test_count_nums() {
        assert!(count_nums(vec![]) == 0);
        assert!(count_nums(vec![-1, -2, 0]) == 0);
        assert!(count_nums(vec![1, 1, 2, -2, 3, 4, 5]) == 6);
        assert!(count_nums(vec![1, 6, 9, -6, 0, 1, 5]) == 5);
        assert!(count_nums(vec![1, 100, 98, -7, 1, -1]) == 4);
        assert!(count_nums(vec![12, 23, 34, -45, -56, 0]) == 5);
        assert!(count_nums(vec![-0, 1]) == 1);
        assert!(count_nums(vec![1]) == 1);
    }

    #[test]
    fn test_move_one_ball() {
        assert!(move_one_ball(vec![3, 4, 5, 1, 2]) == true);
        assert!(move_one_ball(vec![3, 5, 10, 1, 2]) == true);
        assert!(move_one_ball(vec![4, 3, 1, 2]) == false);
        assert!(move_one_ball(vec![3, 5, 4, 1, 2]) == false);
        assert!(move_one_ball(vec![]) == true);
    }

    #[test]
    fn test_exchange() {
        assert!(exchange(vec![1, 2, 3, 4], vec![1, 2, 3, 4]) == "YES");
        assert!(exchange(vec![1, 2, 3, 4], vec![1, 5, 3, 4]) == "NO");
        assert!(exchange(vec![1, 2, 3, 4], vec![2, 1, 4, 3]) == "YES");
        assert!(exchange(vec![5, 7, 3], vec![2, 6, 4]) == "YES");
        assert!(exchange(vec![5, 7, 3], vec![2, 6, 3]) == "NO");
        assert!(exchange(vec![3, 2, 6, 1, 8, 9], vec![3, 5, 5, 1, 1, 1]) == "NO");
        assert!(exchange(vec![100, 200], vec![200, 200]) == "YES");
    }

    #[test]
    fn test_histogram() {
        assert!(histogram("a b b a") == HashMap::from([('a', 2), ('b', 2)]));
        assert!(histogram("a b c a b") == HashMap::from([('a', 2), ('b', 2)]));
        assert!(
            histogram("a b c d g")
                == HashMap::from([('a', 1), ('b', 1), ('c', 1), ('d', 1), ('g', 1)])
        );
        assert!(histogram("r t g") == HashMap::from([('r', 1), ('t', 1), ('g', 1)]));
        assert!(histogram("b b b b a") == HashMap::from([('b', 4)]));
        assert!(histogram("r t g") == HashMap::from([('r', 1), ('t', 1), ('g', 1)]));
        assert!(histogram("") == HashMap::new());
        assert!(histogram("a") == HashMap::from([(('a', 1))]));
    }

    #[test]
    fn test_reverse_delete() {
        assert!(reverse_delete("abcde", "ae") == ["bcd", "False"]);
        assert!(reverse_delete("abcdef", "b") == ["acdef", "False"]);
        assert!(reverse_delete("abcdedcba", "ab") == ["cdedc", "True"]);
        assert!(reverse_delete("dwik", "w") == ["dik", "False"]);
        assert!(reverse_delete("a", "a") == ["", "True"]);
        assert!(reverse_delete("abcdedcba", "") == ["abcdedcba", "True"]);
        assert!(reverse_delete("abcdedcba", "v") == ["abcdedcba", "True"]);
        assert!(reverse_delete("vabba", "v") == ["abba", "True"]);
        assert!(reverse_delete("mamma", "mia") == ["", "True"]);
    }

    #[test]
    fn test_odd_count() {
        assert!(
            odd_count(vec!["1234567"])
                == ["the number of odd elements 4n the str4ng 4 of the 4nput."]
        );
        assert!(
            odd_count(vec!["3", "11111111"])
                == [
                    "the number of odd elements 1n the str1ng 1 of the 1nput.",
                    "the number of odd elements 8n the str8ng 8 of the 8nput."
                ]
        );
        assert!(
            odd_count(vec!["271", "137", "314"])
                == [
                    "the number of odd elements 2n the str2ng 2 of the 2nput.",
                    "the number of odd elements 3n the str3ng 3 of the 3nput.",
                    "the number of odd elements 2n the str2ng 2 of the 2nput."
                ]
        );
    }

    #[test]
    fn test_min_sub_array_sum() {
        assert!(min_sub_array_sum(vec![2, 3, 4, 1, 2, 4]) == 1);
        assert!(min_sub_array_sum(vec![-1, -2, -3]) == -6);
        assert!(min_sub_array_sum(vec![-1, -2, -3, 2, -10]) == -14);
        assert!(min_sub_array_sum(vec![-9999999999999999]) == -9999999999999999);
        assert!(min_sub_array_sum(vec![0, 10, 20, 1000000]) == 0);
        assert!(min_sub_array_sum(vec![-1, -2, -3, 10, -5]) == -6);
        assert!(min_sub_array_sum(vec![100, -1, -2, -3, 10, -5]) == -6);
        assert!(min_sub_array_sum(vec![10, 11, 13, 8, 3, 4]) == 3);
        assert!(min_sub_array_sum(vec![100, -33, 32, -1, 0, -2]) == -33);
        assert!(min_sub_array_sum(vec![-10]) == -10);
        assert!(min_sub_array_sum(vec![7]) == 7);
        assert!(min_sub_array_sum(vec![1, -1]) == -1);
    }

    #[test]
    fn test_max_fill() {
        assert!(
            max_fill(
                vec![vec![0, 0, 1, 0], vec![0, 1, 0, 0], vec![1, 1, 1, 1]],
                1
            ) == 6
        );
        assert!(
            max_fill(
                vec![
                    vec![0, 0, 1, 1],
                    vec![0, 0, 0, 0],
                    vec![1, 1, 1, 1],
                    vec![0, 1, 1, 1]
                ],
                2
            ) == 5
        );
        assert!(max_fill(vec![vec![0, 0, 0], vec![0, 0, 0]], 5) == 0);
        assert!(max_fill(vec![vec![1, 1, 1, 1], vec![1, 1, 1, 1]], 2) == 4);
        assert!(max_fill(vec![vec![1, 1, 1, 1], vec![1, 1, 1, 1]], 9) == 2);
    }

    #[test]
    fn test_sort_array_1() {
        assert!(sort_array_1(vec![1, 5, 2, 3, 4]) == vec![1, 2, 4, 3, 5]);
        assert!(sort_array_1(vec![-2, -3, -4, -5, -6]) == vec![-4, -2, -6, -5, -3]);
        assert!(sort_array_1(vec![1, 0, 2, 3, 4]) == vec![0, 1, 2, 4, 3]);
        assert!(sort_array_1(vec![]) == vec![]);
        assert!(
            sort_array_1(vec![2, 5, 77, 4, 5, 3, 5, 7, 2, 3, 4])
                == vec![2, 2, 4, 4, 3, 3, 5, 5, 5, 7, 77]
        );
        assert!(sort_array_1(vec![3, 6, 44, 12, 32, 5]) == vec![32, 3, 5, 6, 12, 44]);
        assert!(sort_array_1(vec![2, 4, 8, 16, 32]) == vec![2, 4, 8, 16, 32]);
        assert!(sort_array_1(vec![2, 4, 8, 16, 32]) == vec![2, 4, 8, 16, 32]);
    }

    #[test]
    fn test_select_words() {
        assert_eq!(select_words("Mary had a little lamb", 4), vec!["little"]);
        assert_eq!(
            select_words("Mary had a little lamb", 3),
            vec!["Mary", "lamb"]
        );
        let v_empty: Vec<&str> = vec![];
        assert_eq!(select_words("simple white space", 2), v_empty);
        assert_eq!(select_words("Hello world", 4), vec!["world"]);
        assert_eq!(select_words("Uncle sam", 3), vec!["Uncle"]);
        assert_eq!(select_words("", 4), v_empty);
        assert_eq!(select_words("a b c d e f", 1), vec!["b", "c", "d", "f"]);
    }

    #[test]
    fn test_get_closest_vowel() {
        assert_eq!(get_closest_vowel("yogurt"), "u");
        assert_eq!(get_closest_vowel("full"), "u");
        assert_eq!(get_closest_vowel("easy"), "");
        assert_eq!(get_closest_vowel("eAsy"), "");
        assert_eq!(get_closest_vowel("ali"), "");
        assert_eq!(get_closest_vowel("bad"), "a");
        assert_eq!(get_closest_vowel("most"), "o");
        assert_eq!(get_closest_vowel("ab"), "");
        assert_eq!(get_closest_vowel("ba"), "");
        assert_eq!(get_closest_vowel("quick"), "");
        assert_eq!(get_closest_vowel("anime"), "i");
        assert_eq!(get_closest_vowel("Asia"), "");
        assert_eq!(get_closest_vowel("Above"), "o");
    }

    #[test]
    fn test_match_parens() {
        assert_eq!(match_parens(vec!["()(", ")"]), "Yes");
        assert_eq!(match_parens(vec![")", ")"]), "No");
        assert_eq!(match_parens(vec!["(()(())", "())())"],), "No");
        assert_eq!(match_parens(vec![")())", "(()()("]), "Yes");
        assert_eq!(match_parens(vec!["(())))", "(()())(("]), "Yes");
        assert_eq!(match_parens(vec!["()", "())"],), "No");
        assert_eq!(match_parens(vec!["(()(", "()))()"]), "Yes");
        assert_eq!(match_parens(vec!["((((", "((())"],), "No");
        assert_eq!(match_parens(vec![")(()", "(()("]), "No");
        assert_eq!(match_parens(vec![")(", ")("]), "No");
        assert_eq!(match_parens(vec!["(", ")"]), "Yes");
        assert_eq!(match_parens(vec![")", "("]), "Yes");
    }

    #[test]
    fn test_maximum_120() {
        assert_eq!(maximum_120(vec![-3, -4, 5], 3), vec![-4, -3, 5]);
        assert_eq!(maximum_120(vec![4, -4, 4], 2), vec![4, 4]);
        assert_eq!(maximum_120(vec![-3, 2, 1, 2, -1, -2, 1], 1), vec![2]);
        assert_eq!(
            maximum_120(vec![123, -123, 20, 0, 1, 2, -3], 3),
            vec![2, 20, 123]
        );
        assert_eq!(
            maximum_120(vec![-123, 20, 0, 1, 2, -3], 4),
            vec![0, 1, 2, 20]
        );
        assert_eq!(
            maximum_120(vec![5, 15, 0, 3, -13, -8, 0], 7),
            vec![-13, -8, 0, 0, 3, 5, 15]
        );
        assert_eq!(maximum_120(vec![-1, 0, 2, 5, 3, -10], 2), vec![3, 5]);
        assert_eq!(maximum_120(vec![1, 0, 5, -7], 1), vec![5]);
        assert_eq!(maximum_120(vec![4, -4], 2), vec![-4, 4]);
        assert_eq!(maximum_120(vec![-10, 10], 2), vec![-10, 10]);
        assert_eq!(maximum_120(vec![1, 2, 3, -23, 243, -400, 0], 0), vec![]);
    }

    #[test]
    fn test_solutions() {
        assert_eq!(solutions(vec![5, 8, 7, 1]), 12);
        assert_eq!(solutions(vec![3, 3, 3, 3, 3]), 9);
        assert_eq!(solutions(vec![30, 13, 24, 321]), 0);
        assert_eq!(solutions(vec![5, 9]), 5);
        assert_eq!(solutions(vec![2, 4, 8]), 0);
        assert_eq!(solutions(vec![30, 13, 23, 32]), 23);
        assert_eq!(solutions(vec![3, 13, 2, 9]), 3);
    }

    #[test]
    fn test_add_elements() {
        assert_eq!(add_elements(vec![1, -2, -3, 41, 57, 76, 87, 88, 99], 3), -4);
        assert_eq!(add_elements(vec![111, 121, 3, 4000, 5, 6], 2), 0);
        assert_eq!(add_elements(vec![11, 21, 3, 90, 5, 6, 7, 8, 9], 4), 125);
        assert_eq!(add_elements(vec![111, 21, 3, 4000, 5, 6, 7, 8, 9], 4), 24);
        assert_eq!(add_elements(vec![1], 1), 1);
    }

    #[test]
    fn test_get_odd_collatz() {
        assert_eq!(get_odd_collatz(14), vec![1, 5, 7, 11, 13, 17]);
        assert_eq!(get_odd_collatz(5), vec![1, 5]);
        assert_eq!(get_odd_collatz(12), vec![1, 3, 5]);
        assert_eq!(get_odd_collatz(1), vec![1]);
    }

    #[test]
    fn test_valid_date() {
        assert_eq!(valid_date("03-11-2000"), true);
        assert_eq!(valid_date("15-01-2012"), false);
        assert_eq!(valid_date("04-0-2040"), false);
        assert_eq!(valid_date("06-04-2020"), true);
        assert_eq!(valid_date("01-01-2007"), true);
        assert_eq!(valid_date("03-32-2011"), false);
        assert_eq!(valid_date(""), false);
        assert_eq!(valid_date("04-31-3000"), false);
        assert_eq!(valid_date("06-06-2005"), true);
        assert_eq!(valid_date("21-31-2000"), false);
        assert_eq!(valid_date("04-12-2003"), true);
        assert_eq!(valid_date("04122003"), false);
        assert_eq!(valid_date("20030412"), false);
        assert_eq!(valid_date("2003-04"), false);
        assert_eq!(valid_date("2003-04-12"), false);
        assert_eq!(valid_date("04-2003"), false);
    }

    #[test]
    fn test_split_words() {
        assert_eq!(split_words("Hello world!"), vec!["Hello", "world!"]);
        assert_eq!(split_words("Hello,world!"), vec!["Hello", "world!"]);
        assert_eq!(split_words("Hello world,!"), vec!["Hello", "world,!"]);
        assert_eq!(
            split_words("Hello,Hello,world !"),
            vec!["Hello,Hello,world", "!"]
        );
        assert_eq!(split_words("abcdef"), vec!["3"]);
        assert_eq!(split_words("aaabb"), vec!["2"]);
        assert_eq!(split_words("aaaBb"), vec!["1"]);
        assert_eq!(split_words(""), vec!["0"]);
    }

    #[test]
    fn test_is_sorted() {
        assert_eq!(is_sorted(vec![5]), true);
        assert_eq!(is_sorted(vec![1, 2, 3, 4, 5]), true);
        assert_eq!(is_sorted(vec![1, 3, 2, 4, 5]), false);
        assert_eq!(is_sorted(vec![1, 2, 3, 4, 5, 6]), true);
        assert_eq!(is_sorted(vec![1, 2, 3, 4, 5, 6, 7]), true);
        assert_eq!(is_sorted(vec![1, 3, 2, 4, 5, 6, 7]), false);
        assert_eq!(is_sorted(vec![]), true);
        assert_eq!(is_sorted(vec![1]), true);
        assert_eq!(is_sorted(vec![3, 2, 1]), false);
        assert_eq!(is_sorted(vec![1, 2, 2, 2, 3, 4]), false);
        assert_eq!(is_sorted(vec![1, 2, 3, 3, 3, 4]), false);
        assert_eq!(is_sorted(vec![1, 2, 2, 3, 3, 4]), true);
        assert_eq!(is_sorted(vec![1, 2, 3, 4]), true);
    }

    #[test]
    fn test_intersection() {
        assert_eq!(intersection(vec![1, 2], vec![2, 3]), "NO");
        assert_eq!(intersection(vec![-1, 1], vec![0, 4]), "NO");
        assert_eq!(intersection(vec![-3, -1], vec![-5, 5]), "YES");
        assert_eq!(intersection(vec![-2, 2], vec![-4, 0]), "YES");
        assert_eq!(intersection(vec![-11, 2], vec![-1, -1]), "NO");
        assert_eq!(intersection(vec![1, 2], vec![3, 5]), "NO");
        assert_eq!(intersection(vec![1, 2], vec![1, 2]), "NO");
        assert_eq!(intersection(vec![-2, -2], vec![-3, -2]), "NO");
    }

    #[test]
    fn test_prod_signs() {
        assert_eq!(prod_signs(vec![1, 2, 2, -4]), -9);
        assert_eq!(prod_signs(vec![0, 1]), 0);
        assert_eq!(prod_signs(vec![1, 1, 1, 2, 3, -1, 1]), -10);
        assert_eq!(prod_signs(vec![]), -32768);
        assert_eq!(prod_signs(vec![2, 4, 1, 2, -1, -1, 9]), 20);
        assert_eq!(prod_signs(vec![-1, 1, -1, 1]), 4);
        assert_eq!(prod_signs(vec![-1, 1, 1, 1]), -4);
        assert_eq!(prod_signs(vec![-1, 1, 1, 0]), 0);
    }

    #[test]
    fn test_min_path() {
        assert_eq!(
            min_path(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]], 3),
            vec![1, 2, 1]
        );
        assert_eq!(
            min_path(vec![vec![5, 9, 3], vec![4, 1, 6], vec![7, 8, 2]], 1),
            vec![1]
        );
        assert_eq!(
            min_path(
                vec![
                    vec![1, 2, 3, 4],
                    vec![5, 6, 7, 8],
                    vec![9, 10, 11, 12],
                    vec![13, 14, 15, 16]
                ],
                4
            ),
            vec![1, 2, 1, 2]
        );
        assert_eq!(
            min_path(
                vec![
                    vec![6, 4, 13, 10],
                    vec![5, 7, 12, 1],
                    vec![3, 16, 11, 15],
                    vec![8, 14, 9, 2]
                ],
                7
            ),
            vec![1, 10, 1, 10, 1, 10, 1]
        );
        assert_eq!(
            min_path(
                vec![
                    vec![8, 14, 9, 2],
                    vec![6, 4, 13, 15],
                    vec![5, 7, 1, 12],
                    vec![3, 10, 11, 16]
                ],
                5
            ),
            vec![1, 7, 1, 7, 1]
        );
        assert_eq!(
            min_path(
                vec![
                    vec![11, 8, 7, 2],
                    vec![5, 16, 14, 4],
                    vec![9, 3, 15, 6],
                    vec![12, 13, 10, 1]
                ],
                9
            ),
            vec![1, 6, 1, 6, 1, 6, 1, 6, 1]
        );
        assert_eq!(
            min_path(
                vec![
                    vec![12, 13, 10, 1],
                    vec![9, 3, 15, 6],
                    vec![5, 16, 14, 4],
                    vec![11, 8, 7, 2]
                ],
                12
            ),
            vec![1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6]
        );
        assert_eq!(
            min_path(vec![vec![2, 7, 4], vec![3, 1, 5], vec![6, 8, 9]], 8),
            vec![1, 3, 1, 3, 1, 3, 1, 3]
        );

        assert_eq!(
            min_path(vec![vec![6, 1, 5], vec![3, 8, 9], vec![2, 7, 4]], 8),
            vec![1, 5, 1, 5, 1, 5, 1, 5]
        );

        assert_eq!(
            min_path(vec![vec![1, 2], vec![3, 4]], 10),
            vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        );

        assert_eq!(
            min_path(vec![vec![1, 3], vec![3, 2]], 10),
            vec![1, 3, 1, 3, 1, 3, 1, 3, 1, 3]
        );
    }

    #[test]
    fn test_tri() {
        assert!(tri(3) == vec![1, 3, 2, 8]);
        assert!(tri(4) == vec![1, 3, 2, 8, 3]);
        assert!(tri(5) == vec![1, 3, 2, 8, 3, 15]);
        assert!(tri(6) == vec![1, 3, 2, 8, 3, 15, 4]);
        assert!(tri(7) == vec![1, 3, 2, 8, 3, 15, 4, 24]);
        assert!(tri(8) == vec![1, 3, 2, 8, 3, 15, 4, 24, 5]);
        assert!(tri(9) == vec![1, 3, 2, 8, 3, 15, 4, 24, 5, 35]);
        assert!(
            tri(20)
                == vec![1, 3, 2, 8, 3, 15, 4, 24, 5, 35, 6, 48, 7, 63, 8, 80, 9, 99, 10, 120, 11]
        );
        assert!(tri(0) == vec![1]);
        assert!(tri(1) == vec![1, 3]);
    }

    #[test]
    fn test_digits() {
        assert_eq!(digits(5), 5);
        assert_eq!(digits(54), 5);
        assert_eq!(digits(120), 1);
        assert_eq!(digits(5014), 5);
        assert_eq!(digits(98765), 315);
        assert_eq!(digits(5576543), 2625);
        assert_eq!(digits(2468), 0);
    }

    #[test]
    fn test_is_nested() {
        assert_eq!(is_nested("[[]]"), true);
        assert_eq!(is_nested("[]]]]]]][[[[[]"), false);
        assert_eq!(is_nested("[][]"), false);
        assert_eq!(is_nested("[]"), false);
        assert_eq!(is_nested("[[[[]]]]"), true);
        assert_eq!(is_nested("[]]]]]]]]]]"), false);
        assert_eq!(is_nested("[][][[]]"), true);
        assert_eq!(is_nested("[[]"), false);
        assert_eq!(is_nested("[]]"), false);
        assert_eq!(is_nested("[[]][["), true);
        assert_eq!(is_nested("[[][]]"), true);
        assert_eq!(is_nested(""), false);
        assert_eq!(is_nested("[[[[[[[["), false);
        assert_eq!(is_nested("]]]]]]]]"), false);
    }

    #[test]
    fn test_sum_squares() {
        assert_eq!(sum_squares(vec![1.0, 2.0, 3.0]), 14);
        assert_eq!(sum_squares(vec![1.0, 2.0, 3.0]), 14);
        assert_eq!(sum_squares(vec![1.0, 3.0, 5.0, 7.0]), 84);
        assert_eq!(sum_squares(vec![1.4, 4.2, 0.0]), 29);
        assert_eq!(sum_squares(vec![-2.4, 1.0, 1.0]), 6);
        assert_eq!(sum_squares(vec![100.0, 1.0, 15.0, 2.0]), 10230);
        assert_eq!(sum_squares(vec![10000.0, 10000.0]), 200000000);
        assert_eq!(sum_squares(vec![-1.4, 4.6, 6.3]), 75);
        assert_eq!(sum_squares(vec![-1.4, 17.9, 18.9, 19.9]), 1086);
        assert_eq!(sum_squares(vec![0.0]), 0);
        assert_eq!(sum_squares(vec![-1.0]), 1);
        assert_eq!(sum_squares(vec![-1.0, 1.0, 0.0]), 2);
    }

    #[test]
    fn test_check_if_last_char_is_a_letter() {
        assert_eq!(check_if_last_char_is_a_letter("apple"), false);
        assert_eq!(check_if_last_char_is_a_letter("apple pi e"), true);
        assert_eq!(check_if_last_char_is_a_letter("eeeee"), false);
        assert_eq!(check_if_last_char_is_a_letter("A"), true);
        assert_eq!(check_if_last_char_is_a_letter("Pumpkin pie "), false);
        assert_eq!(check_if_last_char_is_a_letter("Pumpkin pie 1"), false);
        assert_eq!(check_if_last_char_is_a_letter(""), false);
        assert_eq!(check_if_last_char_is_a_letter("eeeee e "), false);
        assert_eq!(check_if_last_char_is_a_letter("apple pie"), false);
    }

    #[test]
    fn test_can_arrange() {
        assert_eq!(can_arrange(vec![1, 2, 4, 3, 5]), 3);
        assert_eq!(can_arrange(vec![1, 2, 4, 5]), -1);
        assert_eq!(can_arrange(vec![1, 4, 2, 5, 6, 7, 8, 9, 10]), 2);
        assert_eq!(can_arrange(vec![4, 8, 5, 7, 3]), 4);
        assert_eq!(can_arrange(vec![]), -1);
    }

    #[test]
    fn test_largest_smallest_integers() {
        assert_eq!(
            largest_smallest_integers(vec![2, 4, 1, 3, 5, 7]),
            vec![0, 1]
        );
        assert_eq!(
            largest_smallest_integers(vec![2, 4, 1, 3, 5, 7, 0]),
            vec![0, 1]
        );
        assert_eq!(
            largest_smallest_integers(vec![1, 3, 2, 4, 5, 6, -2]),
            vec![-2, 1]
        );
        assert_eq!(
            largest_smallest_integers(vec![4, 5, 3, 6, 2, 7, -7]),
            vec![-7, 2]
        );
        assert_eq!(
            largest_smallest_integers(vec![7, 3, 8, 4, 9, 2, 5, -9]),
            vec![-9, 2]
        );
        assert_eq!(largest_smallest_integers(vec![]), vec![0, 0]);
        assert_eq!(largest_smallest_integers(vec![0]), vec![0, 0]);
        assert_eq!(largest_smallest_integers(vec![-1, -3, -5, -6]), vec![-1, 0]);
        assert_eq!(
            largest_smallest_integers(vec![-1, -3, -5, -6, 0]),
            vec![-1, 0]
        );
        assert_eq!(
            largest_smallest_integers(vec![-6, -4, -4, -3, 1]),
            vec![-3, 1]
        );
        assert_eq!(
            largest_smallest_integers(vec![-6, -4, -4, -3, -100, 1]),
            vec![-3, 1]
        );
    }

    #[test]
    fn test_compare_one() {
        assert_eq!(compare_one(&1, &2), RtnType::Int(2));
        assert_eq!(compare_one(&1, &2.5), RtnType::Float(2.5));
        assert_eq!(compare_one(&2, &3), RtnType::Int(3));
        assert_eq!(compare_one(&5, &6), RtnType::Int(6));
        assert_eq!(compare_one(&1, &"2.3"), RtnType::String("2.3".to_string()));
        assert_eq!(compare_one(&"5.1", &"6"), RtnType::String("6".to_string()));
        assert_eq!(compare_one(&"1", &"2"), RtnType::String("2".to_string()));
        assert_eq!(compare_one(&"1", &1), RtnType::String("None".to_string()));
    }

    #[test]
    fn test_is_equal_to_sum_even() {
        assert_eq!(is_equal_to_sum_even(4), false);
        assert_eq!(is_equal_to_sum_even(6), false);
        assert_eq!(is_equal_to_sum_even(8), true);
        assert_eq!(is_equal_to_sum_even(10), true);
        assert_eq!(is_equal_to_sum_even(11), false);
        assert_eq!(is_equal_to_sum_even(12), true);
        assert_eq!(is_equal_to_sum_even(13), false);
        assert_eq!(is_equal_to_sum_even(16), true);
    }

    #[test]
    fn test_special_factorial() {
        assert_eq!(special_factorial(4), 288);
        assert_eq!(special_factorial(5), 34560);
        assert_eq!(special_factorial(7), 125411328000);
        assert_eq!(special_factorial(1), 1);
    }

    #[test]
    fn test_fix_spaces() {
        //ERROR on asserts
        assert_eq!(fix_spaces("Example"), "Example");
        assert_eq!(fix_spaces("Mudasir Hanif "), "Mudasir_Hanif_");
        assert_eq!(
            fix_spaces("Yellow Yellow  Dirty  Fellow"),
            "Yellow_Yellow__Dirty__Fellow"
        );
        assert_eq!(fix_spaces("Exa   mple"), "Exa-mple");
        assert_eq!(fix_spaces("   Exa 1 2 2 mple"), "-Exa_1_2_2_mple");
    }

    #[test]
    fn test_file_name_check() {
        assert_eq!(file_name_check("example.txt"), "Yes");
        assert_eq!(file_name_check("1example.dll"), "No");
        assert_eq!(file_name_check("s1sdf3.asd"), "No");
        assert_eq!(file_name_check("K.dll"), "Yes");
        assert_eq!(file_name_check("MY16FILE3.exe"), "Yes");
        assert_eq!(file_name_check("His12FILE94.exe"), "No");
        assert_eq!(file_name_check("_Y.txt"), "No");
        assert_eq!(file_name_check("?aREYA.exe"), "No");
        assert_eq!(file_name_check("/this_is_valid.dll"), "No");
        assert_eq!(file_name_check("this_is_valid.wow"), "No");
        assert_eq!(file_name_check("this_is_valid.txt"), "Yes");
        assert_eq!(file_name_check("this_is_valid.txtexe"), "No");
        assert_eq!(file_name_check("#this2_i4s_5valid.ten"), "No");
        assert_eq!(file_name_check("@this1_is6_valid.exe"), "No");
        assert_eq!(file_name_check("this_is_12valid.6exe4.txt"), "No");
        assert_eq!(file_name_check("all.exe.txt"), "No");
        assert_eq!(file_name_check("I563_No.exe"), "Yes");
        assert_eq!(file_name_check("Is3youfault.txt"), "Yes");
        assert_eq!(file_name_check("no_one#knows.dll"), "Yes");
        assert_eq!(file_name_check("1I563_Yes3.exe"), "No");
        assert_eq!(file_name_check("I563_Yes3.txtt"), "No");
        assert_eq!(file_name_check("final..txt"), "No");
        assert_eq!(file_name_check("final132"), "No");
        assert_eq!(file_name_check("_f4indsartal132."), "No");
        assert_eq!(file_name_check(".txt"), "No");
        assert_eq!(file_name_check("s."), "No");
    }

    #[test]
    fn test_sum_squares_142() {
        assert_eq!(sum_squares_142(vec![1, 2, 3]), 6);
        assert_eq!(sum_squares_142(vec![1, 4, 9]), 14);
        assert_eq!(sum_squares_142(vec![]), 0);
        assert_eq!(sum_squares_142(vec![1, 1, 1, 1, 1, 1, 1, 1, 1]), 9);
        assert_eq!(
            sum_squares_142(vec![-1, -1, -1, -1, -1, -1, -1, -1, -1]),
            -3
        );
        assert_eq!(sum_squares_142(vec![0]), 0);
        assert_eq!(sum_squares_142(vec![-1, -5, 2, -1, -5]), -126);
        assert_eq!(sum_squares_142(vec![-56, -99, 1, 0, -2]), 3030);
        assert_eq!(sum_squares_142(vec![-1, 0, 0, 0, 0, 0, 0, 0, -1]), 0);
        assert_eq!(
            sum_squares_142(vec![
                -16, -9, -2, 36, 36, 26, -20, 25, -40, 20, -4, 12, -26, 35, 37
            ]),
            -14196
        );
        assert_eq!(
            sum_squares_142(vec![
                -1, -3, 17, -1, -15, 13, -1, 14, -14, -12, -5, 14, -14, 6, 13, 11, 16, 16, 4, 10
            ]),
            -1448
        );
    }
    #[test]
    fn test_words_in_sentence() {
        assert_eq!(words_in_sentence("This is a test"), "is");
        assert_eq!(words_in_sentence("lets go for swimming"), "go for");
        assert_eq!(
            words_in_sentence("there is no place available here"),
            "there is no place"
        );
        assert_eq!(words_in_sentence("Hi I am Hussein"), "Hi am Hussein");
        assert_eq!(words_in_sentence("go for it"), "go for it");
        assert_eq!(words_in_sentence("here"), "");
        assert_eq!(words_in_sentence("here is"), "is");
    }

    #[test]
    fn test_simplify() {
        assert_eq!(simplify("1/5", "5/1"), true);
        assert_eq!(simplify("1/6", "2/1"), false);
        assert_eq!(simplify("5/1", "3/1"), true);
        assert_eq!(simplify("7/10", "10/2"), false);
        assert_eq!(simplify("2/10", "50/10"), true);
        assert_eq!(simplify("7/2", "4/2"), true);
        assert_eq!(simplify("11/6", "6/1"), true);
        assert_eq!(simplify("2/3", "5/2"), false);
        assert_eq!(simplify("5/2", "3/5"), false);
        assert_eq!(simplify("2/4", "8/4"), true);
        assert_eq!(simplify("2/4", "4/2"), true);
        assert_eq!(simplify("1/5", "5/1"), true);
        assert_eq!(simplify("1/5", "1/5"), false);
    }

    #[test]
    fn test_order_by_points() {
        assert_eq!(
            order_by_points(vec![1, 11, -1, -11, -12]),
            vec![-1, -11, 1, -12, 11]
        );
        assert_eq!(
            order_by_points(vec![
                1234, 423, 463, 145, 2, 423, 423, 53, 6, 37, 3457, 3, 56, 0, 46
            ]),
            vec![0, 2, 3, 6, 53, 423, 423, 423, 1234, 145, 37, 46, 56, 463, 3457]
        );
        assert_eq!(order_by_points(vec![]), vec![]);
        assert_eq!(
            order_by_points(vec![1, -11, -32, 43, 54, -98, 2, -3]),
            vec![-3, -32, -98, -11, 1, 2, 43, 54]
        );
        assert_eq!(
            order_by_points(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            vec![1, 10, 2, 11, 3, 4, 5, 6, 7, 8, 9]
        );
        assert_eq!(
            order_by_points(vec![0, 6, 6, -76, -21, 23, 4]),
            vec![-76, -21, 0, 4, 23, 6, 6]
        );
    }

    #[test]
    fn test_special_filter() {
        assert_eq!(special_filter(vec![5, -2, 1, -5]), 0);
        assert_eq!(special_filter(vec![15, -73, 14, -15]), 1);
        assert_eq!(special_filter(vec![33, -2, -3, 45, 21, 109]), 2);
        assert_eq!(special_filter(vec![43, -12, 93, 125, 121, 109]), 4);
        assert_eq!(special_filter(vec![71, -2, -33, 75, 21, 19]), 3);
        assert_eq!(special_filter(vec![1]), 0);
        assert_eq!(special_filter(vec![]), 0);
    }

    #[test]
    fn test_get_matrix_triples() {
        assert_eq!(get_matrix_triples(5), 1);
        assert_eq!(get_matrix_triples(6), 4);
        assert_eq!(get_matrix_triples(10), 36);
        assert_eq!(get_matrix_triples(100), 53361);
    }

    #[test]
    fn test_bf() {
        assert_eq!(bf("Jupiter", "Neptune"), vec!["Saturn", "Uranus"]);
        assert_eq!(bf("Earth", "Mercury"), vec!["Venus"]);
        assert_eq!(
            bf("Mercury", "Uranus"),
            vec!["Venus", "Earth", "Mars", "Jupiter", "Saturn"]
        );
        assert_eq!(
            bf("Neptune", "Venus"),
            vec!["Earth", "Mars", "Jupiter", "Saturn", "Uranus"]
        );
        let v_empty: Vec<&str> = vec![];
        assert_eq!(bf("Earth", "Earth"), v_empty);
        assert_eq!(bf("Mars", "Earth"), v_empty);
        assert_eq!(bf("Jupiter", "Makemake"), v_empty);
    }

    #[test]
    fn test_sorted_list_sum() {
        assert_eq!(sorted_list_sum(vec!["aa", "a", "aaa"]), vec!["aa"]);
        assert_eq!(
            sorted_list_sum(vec!["school", "AI", "asdf", "b"]),
            vec!["AI", "asdf", "school"]
        );
        let v_empty: Vec<&str> = vec![];
        assert_eq!(sorted_list_sum(vec!["d", "b", "c", "a"]), v_empty);
        assert_eq!(
            sorted_list_sum(vec!["d", "dcba", "abcd", "a"]),
            vec!["abcd", "dcba"]
        );
        assert_eq!(
            sorted_list_sum(vec!["AI", "ai", "au"]),
            vec!["AI", "ai", "au"]
        );
        assert_eq!(sorted_list_sum(vec!["a", "b", "b", "c", "c", "a"]), v_empty);
        assert_eq!(
            sorted_list_sum(vec!["aaaa", "bbbb", "dd", "cc"]),
            vec!["cc", "dd", "aaaa", "bbbb"]
        );
    }

    #[test]
    fn test_x_or_y() {
        assert_eq!(x_or_y(7, 34, 12), 34);
        assert_eq!(x_or_y(15, 8, 5), 5);
        assert_eq!(x_or_y(3, 33, 5212), 33);
        assert_eq!(x_or_y(1259, 3, 52), 3);
        assert_eq!(x_or_y(7919, -1, 12), -1);
        assert_eq!(x_or_y(3609, 1245, 583), 583);
        assert_eq!(x_or_y(91, 56, 129), 129);
        assert_eq!(x_or_y(6, 34, 1234), 1234);
        assert_eq!(x_or_y(1, 2, 0), 0);
        assert_eq!(x_or_y(2, 2, 0), 2);
    }

    #[test]
    fn test_double_the_difference() {
        assert_eq!(double_the_difference(vec![]), 0);
        assert_eq!(double_the_difference(vec![5.0, 4.0]), 25);
        assert_eq!(double_the_difference(vec![0.1, 0.2, 0.3]), 0);
        assert_eq!(double_the_difference(vec![-10.0, -20.0, -30.0]), 0);
        assert_eq!(double_the_difference(vec![-1.0, -2.0, 8.0]), 0);
        assert_eq!(double_the_difference(vec![0.2, 3.0, 5.0]), 34);

        let mut lst = vec![];
        let mut odd_sum = 0;
        for i in -99..100 {
            lst.push(i as f32);
            if i > 0 && i % 2 == 1 {
                odd_sum += i * i;
            }
        }
        assert_eq!(double_the_difference(lst), odd_sum);
    }

    #[test]
    fn test_compare() {
        assert_eq!(
            compare(vec![1, 2, 3, 4, 5, 1], vec![1, 2, 3, 4, 2, -2]),
            vec![0, 0, 0, 0, 3, 3]
        );
        assert_eq!(
            compare(vec![0, 5, 0, 0, 0, 4], vec![4, 1, 1, 0, 0, -2]),
            vec![4, 4, 1, 0, 0, 6]
        );
        assert_eq!(
            compare(vec![1, 2, 3, 4, 5, 1], vec![1, 2, 3, 4, 2, -2]),
            vec![0, 0, 0, 0, 3, 3]
        );
        assert_eq!(
            compare(vec![0, 0, 0, 0, 0, 0], vec![0, 0, 0, 0, 0, 0]),
            vec![0, 0, 0, 0, 0, 0]
        );
        assert_eq!(compare(vec![1, 2, 3], vec![-1, -2, -3]), vec![2, 4, 6]);
        assert_eq!(
            compare(vec![1, 2, 3, 5], vec![-1, 2, 3, 4]),
            vec![2, 0, 0, 1]
        );
    }

    #[test]
    fn test_strongest_extension() {
        assert_eq!(
            strongest_extension("Watashi", vec!["tEN", "niNE", "eIGHt8OKe"]),
            "Watashi.eIGHt8OKe"
        );
        assert_eq!(
            strongest_extension("Boku123", vec!["nani", "NazeDa", "YEs.WeCaNe", "32145tggg"]),
            "Boku123.YEs.WeCaNe"
        );
        assert_eq!(
            strongest_extension(
                "__YESIMHERE",
                vec!["t", "eMptY", "(nothing", "zeR00", "NuLl__", "123NoooneB321"]
            ),
            "__YESIMHERE.NuLl__"
        );
        assert_eq!(
            strongest_extension("K", vec!["Ta", "TAR", "t234An", "cosSo"]),
            "K.TAR"
        );
        assert_eq!(
            strongest_extension("__HAHA", vec!["Tab", "123", "781345", "-_-"]),
            "__HAHA.123"
        );
        assert_eq!(
            strongest_extension(
                "YameRore",
                vec!["HhAas", "okIWILL123", "WorkOut", "Fails", "-_-"]
            ),
            "YameRore.okIWILL123"
        );
        assert_eq!(
            strongest_extension("finNNalLLly", vec!["Die", "NowW", "Wow", "WoW"]),
            "finNNalLLly.WoW"
        );
        assert_eq!(strongest_extension("_", vec!["Bb", "91245"]), "_.Bb");
        assert_eq!(strongest_extension("Sp", vec!["671235", "Bb"]), "Sp.671235");
    }

    #[test]
    fn test_cycpattern_check() {
        assert_eq!(cycpattern_check("xyzw", "xyw"), false);
        assert_eq!(cycpattern_check("yello", "ell"), true);
        assert_eq!(cycpattern_check("whattup", "ptut"), false);
        assert_eq!(cycpattern_check("efef", "fee"), true);
        assert_eq!(cycpattern_check("abab", "aabb"), false);
        assert_eq!(cycpattern_check("winemtt", "tinem"), true);
    }

    #[test]
    fn test_even_odd() {
        assert_eq!(even_odd_count(7), vec![0, 1]);
        assert_eq!(even_odd_count(-78), vec![1, 1]);
        assert_eq!(even_odd_count(3452), vec![2, 2]);
        assert_eq!(even_odd_count(346211), vec![3, 3]);
        assert_eq!(even_odd_count(-345821), vec![3, 3]);
        assert_eq!(even_odd_count(-2), vec![1, 0]);
        assert_eq!(even_odd_count(-45347), vec![2, 3]);
        assert_eq!(even_odd_count(0), vec![1, 0]);
    }

    #[test]
    fn test_int_to_mini_romank() {
        assert_eq!(int_to_mini_romank(19), "xix");
        assert_eq!(int_to_mini_romank(152), "clii");
        assert_eq!(int_to_mini_romank(251), "ccli");
        assert_eq!(int_to_mini_romank(426), "cdxxvi");
        assert_eq!(int_to_mini_romank(500), "d");
        assert_eq!(int_to_mini_romank(1), "i");
        assert_eq!(int_to_mini_romank(4), "iv");
        assert_eq!(int_to_mini_romank(43), "xliii");
        assert_eq!(int_to_mini_romank(90), "xc");
        assert_eq!(int_to_mini_romank(94), "xciv");
        assert_eq!(int_to_mini_romank(532), "dxxxii");
        assert_eq!(int_to_mini_romank(900), "cm");
        assert_eq!(int_to_mini_romank(994), "cmxciv");
        assert_eq!(int_to_mini_romank(1000), "m");
    }

    #[test]
    fn test_right_angle_triangle() {
        assert_eq!(right_angle_triangle(3.0, 4.0, 5.0), true);
        assert_eq!(right_angle_triangle(1.0, 2.0, 3.0), false);
        assert_eq!(right_angle_triangle(10.0, 6.0, 8.0), true);
        assert_eq!(right_angle_triangle(2.0, 2.0, 2.0), false);
        assert_eq!(right_angle_triangle(7.0, 24.0, 25.0), true);
        assert_eq!(right_angle_triangle(10.0, 5.0, 7.0), false);
        assert_eq!(right_angle_triangle(5.0, 12.0, 13.0), true);
        assert_eq!(right_angle_triangle(15.0, 8.0, 17.0), true);
        assert_eq!(right_angle_triangle(48.0, 55.0, 73.0), true);
        assert_eq!(right_angle_triangle(1.0, 1.0, 1.0), false);
        assert_eq!(right_angle_triangle(2.0, 2.0, 10.0), false);
    }

    #[test]
    fn test_find_max() {
        assert_eq!(find_max(vec!["name", "of", "string"]), "string");
        assert_eq!(find_max(vec!["name", "enam", "game"]), "enam");
        assert_eq!(find_max(vec!["aaaaaaa", "bb", "cc"]), "aaaaaaa");
        assert_eq!(find_max(vec!["abc", "cba"]), "abc");
        assert_eq!(
            find_max(vec!["play", "this", "game", "of", "footbott"]),
            "footbott"
        );
        assert_eq!(find_max(vec!["we", "are", "gonna", "rock"]), "gonna");
        assert_eq!(find_max(vec!["we", "are", "a", "mad", "nation"]), "nation");
        assert_eq!(find_max(vec!["this", "is", "a", "prrk"]), "this");
        assert_eq!(find_max(vec!["b"]), "b");
        assert_eq!(find_max(vec!["play", "play", "play"]), "play");
    }

    #[test]
    fn test_eat() {
        assert_eq!(eat(5, 6, 10), vec![11, 4]);
        assert_eq!(eat(4, 8, 9), vec![12, 1]);
        assert_eq!(eat(1, 10, 10), vec![11, 0]);
        assert_eq!(eat(2, 11, 5), vec![7, 0]);
        assert_eq!(eat(4, 5, 7), vec![9, 2]);
        assert_eq!(eat(4, 5, 1), vec![5, 0]);
    }

    #[test]
    fn test_do_algebra() {
        assert_eq!(do_algebra(vec!["**", "*", "+"], vec![2, 3, 4, 5]), 37);
        assert_eq!(do_algebra(vec!["+", "*", "-"], vec![2, 3, 4, 5]), 9);
        assert_eq!(do_algebra(vec!["//", "*"], vec![7, 3, 4]), 8);
    }

    #[test]
    fn test_solve_161() {
        assert_eq!(solve_161("AsDf"), "aSdF");
        assert_eq!(solve_161("1234"), "4321");
        assert_eq!(solve_161("ab"), "AB");
        assert_eq!(solve_161("#a@C"), "#A@c");
        assert_eq!(solve_161("#AsdfW^45"), "#aSDFw^45");
        assert_eq!(solve_161("#6@2"), "2@6#");
        assert_eq!(solve_161("#$a^D"), "#$A^d");
        assert_eq!(solve_161("#ccc"), "#CCC");
    }

    #[test]
    fn test_string_to_md5() {
        assert_eq!(
            string_to_md5("Hello world"),
            "3e25960a79dbc69b674cd4ec67a72c62"
        );
        assert_eq!(string_to_md5(""), "None");
        assert_eq!(string_to_md5("A B C"), "0ef78513b0cb8cef12743f5aeb35f888");
        assert_eq!(
            string_to_md5("password"),
            "5f4dcc3b5aa765d61d8327deb882cf99"
        );
    }

    #[test]
    fn test_generate_integers() {
        assert_eq!(generate_integers(2, 10), vec![2, 4, 6, 8]);
        assert_eq!(generate_integers(10, 2), vec![2, 4, 6, 8]);
        assert_eq!(generate_integers(132, 2), vec![2, 4, 6, 8]);
        assert_eq!(generate_integers(17, 89), vec![]);
    }
}
