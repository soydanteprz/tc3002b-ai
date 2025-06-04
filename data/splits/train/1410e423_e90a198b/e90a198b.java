import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.abs;
import static java.lang.System.out;
import java.util.*;
import java.io.*;
import java.math.*;

public class Template {
    
	static int mod = 1000000007;

	public static void main(String[] args){
        FastScanner sc = new FastScanner();
        PrintWriter out = new PrintWriter(System.out);
        int yo = sc.nextInt();
        while (yo-- > 0) {
            int n = sc.nextInt();
            String s = sc.next();
            String t = sc.next();
            int op1 = cal(s,t,n);
            int op2 = helper(s,t,n,'0');
            int op3 = helper(s,t,n,'1');
            int ans = min(min(op1,op2),op3);
            if(ans == Integer.MAX_VALUE){
                out.println("-1");
            }
            else {
                out.println(ans);
            }
        }
        out.close();
	}
    
    static int helper(String str1, String str2, int n, int ch){
        char[] s = str1.toCharArray();
        char[] t = str2.toCharArray();
        int idx = -1;
        for(int i = 0; i < n; i++){
            if(s[i] == '1' && t[i] == ch){
                idx = i;
                break;
            }
        }
        if(idx == -1){
            return Integer.MAX_VALUE;
        }
        for(int i = 0; i < n; i++){
            if(i == idx) continue;
            if(s[i] == '1') s[i] = '0';
            else s[i] = '1';
        }
        int ans = cal(String.valueOf(s),String.valueOf(t),n);
        if(ans == Integer.MAX_VALUE) return ans;
        return 1 + ans;
    }
    static int cal(String s, String t, int n){
        int op01 = 0;
        int op10 = 0;
        for(int i = 0; i < n; i++){
            if(s.charAt(i) != t.charAt(i)){
                if(s.charAt(i) == '1') op10++;
                else op01++;
            }
        }
        if(op10 != op01){
            return Integer.MAX_VALUE;
        }
        return op01 + op10;
    }
    
    /*
    Source: hu_tao
         Random stuff to try when stuck:
            - use bruteforcer
            - check for n = 1, n = 2, so on
            -if it's 2C then it's dp
            -for combo/probability problems, expand the given form we're interested in
            -make everything the same then build an answer (constructive, make everything 0 then do something)
            -something appears in parts of 2 --> model as graph
            -assume a greedy then try to show why it works
            -find way to simplify into one variable if multiple exist
            -treat it like fmc (note any passing thoughts/algo that could be used so you can revisit them)
            -find lower and upper bounds on answer
            -figure out what ur trying to find and isolate it
            -see what observations you have and come up with more continuations
            -work backwards (in constructive, go from the goal to the start)
            -turn into prefix/suffix sum argument (often works if problem revolves around adjacent array elements)
            -instead of solving for answer, try solving for complement (ex, find n-(min) instead of max)
            -draw something
            -simulate a process
            -dont implement something unless if ur fairly confident its correct
            -after 3 bad submissions move on to next problem if applicable
            -do something instead of nothing and stay organized
            -write stuff down
         Random stuff to check when wa:
            -if code is way too long/cancer then reassess
            -switched N/M
            -int overflow
            -switched variables
            -wrong MOD
            -hardcoded edge case incorrectly
         Random stuff to check when tle:
            -continue instead of break
            -condition in for/while loop bad
         Random stuff to check when rte:
            -switched N/M
            -long to int/int overflow
            -division by 0
            -edge case for empty list/data structure/N=1
            
      */

	

	

	

	

	

	
    
    
    
    

	

	//	For Input.txt and Output.txt	
	//	FileInputStream in = new FileInputStream("input.txt");
	//	FileOutputStream out = new FileOutputStream("output.txt");
	//	PrintWriter pw = new PrintWriter(out);
	//	Scanner sc = new Scanner(in);
}
