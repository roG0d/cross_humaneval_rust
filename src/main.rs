pub mod lib;

use cross_humaneval::anti_shuffle;
use cross_humaneval::change_base;
use cross_humaneval::compare_one;
use cross_humaneval::decimal_to_binary;
use cross_humaneval::decode_cyclic;
use cross_humaneval::decode_shift;
use cross_humaneval::encode_cyclic;
use cross_humaneval::encode_shift;
use cross_humaneval::flip_case;
use cross_humaneval::get_row;
use cross_humaneval::next_smallest;
use cross_humaneval::remove_duplicates;
use cross_humaneval::solve;
use cross_humaneval::sort_array;
use cross_humaneval::sort_third;
use cross_humaneval::strange_sort_list;
use cross_humaneval::string_sequence;
use cross_humaneval::sum_of_digits;
use cross_humaneval::total_match;
use cross_humaneval::unique;
use cross_humaneval::words_string;
use lib::has_close_elements;
use lib:: separate_paren_groups;
use lib:: truncate_number;
use lib:: intersperse;
use lib:: parse_nested_parens;
use lib:: parse_music;
use lib :: find_closest_elements;

use crate::lib::all_prefixes;
use crate::lib::fizz_buzz;

fn main() {
    let v1:Vec<f32> = vec![1.0, 2.0, 3.9, 4.0, 5.0, 2.2];
    let t1:f32 = 0.05;
    let r1:bool = has_close_elements(v1, t1);
    println!("{}",r1);

    let s2 = String::from("(()()) ((())) () ((())()())");
    let r2:Vec<String> = separate_paren_groups(s2);
    println!("{:?}",r2);

    let f3: f32 = 3.5;
    let r3: f32 = truncate_number(&f3);
    println!("{}", r3);


    let r4: Vec<u32> = intersperse(vec![2, 2, 2], 2);
    println!("{:?}", r4);

    let r5: Vec<i32> = parse_nested_parens(String::from("(()()) ((())) () ((())()())"));
    println!("{:?}", r5);

    let r6:Vec<String> = all_prefixes(String::from("asdfgh"));
    println!("{:?}", r6);

    let r7:String = string_sequence(10);
    println!("{}",r7);

    let r8:Vec<i32> = parse_music("o o o o".to_string());
    println!("{:?}", r8);

    let r10:f32; let r11:f32;
    (r10, r11) = find_closest_elements(vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.0]);
    println!("{}", r10);
    println!("{}", r11);

    let r12:Vec<i32> = remove_duplicates(vec![1, 2, 3, 2, 4, 3, 5]);
    println!("{:?}", r12);

    let r13 = flip_case("Hello".to_string());
    println!("{}", r13);

    let r14:Vec<i32> = unique(vec![5, 3, 5, 2, 3, 3, 9, 0, 123]);
    println!("{:?}", r14);

    let r15:i32 = fizz_buzz(78);
    println!("{}", r15);

    let r16:String = change_base(9, 3);
    println!("{}", r16);

    let r17:Vec<i32> = strange_sort_list(vec![1, 2, 3, 4]);
    println!("{:?}", r17);

    let r18:Vec<String> = total_match(vec!["hi", "admin"], vec!["hi", "hi"]);
    println!("{:?}", r18);

    let str = vec!["eoeo", "si", "no"];
    let str_count = str.iter().map(|x| x.chars()).count();
    println!("{}", str_count);

    let sqrt_3 =  f64::powf(64.0, 1.0 / 3.0).ceil();
    println!("{}", sqrt_3);

    let str2 = decimal_to_binary(32);
    println!("{}", str2);

    let n  = "150".to_string();
    let str4 = n.to_string().chars().into_iter().fold(0, |acc, c|  acc + c.to_digit(10).unwrap() as i32);
    let str3 = solve(150);
    println!("{}", str3);

    let str5 = anti_shuffle("Hi");
    // "Hello !!!Wdlor");
    println!("{}", str5);

    let r19:Vec<Vec<i32>> = get_row(vec![vec![1,2,3,4,5,6], vec![1,2,3,4,5,6], vec![1,1,3,4,5,6], vec![1,2,1,4,5,6], vec![1,2,3,1,5,6], vec![1,2,3,4,1,6], vec![1,2,3,4,5,1]], 1) ;
    println!("{:?}", r19);

    let r20:Vec<i32> = sort_array(vec![2, 4, 3, 0, 1, 5, 6]);
    println!("{:?}", r20);

    let r21:i32 = next_smallest(vec![1, 2, 3, 4, 5]);
    println!("{}", r21);

    let r22:Vec<String> = words_string("Hi, my name is John");
    println!("{:?}", r22);

    let r23:Vec<i32> = vec![5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10];
    let mut r232:Vec<i32> = vec![];
    

    let s24:String = encode_shift("Hola");
    println!("{}", s24);
    let s25:String = decode_shift(&s24);
    println!("{}", s25);


    let s26:i32 = sum_of_digits(-1);
    println!("{}", s26);

    let r_third = sort_third(vec![5, 8, 3, 4, 6, 9, 2]);
    println!("{:?}", r_third);

    let sth = compare_one(&1.2, &"2");
    println!("{:?}", sth)
} 

