import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.abs;
import static java.lang.System.out;
import java.util.*;
import java.io.*;
import java.math.*;
/*
-> Give your 100%, that's it!
-> Rules To Solve Any Problem:
    1. Read the problem.
    2. Think About It.
    3. Solve it!
*/

public class Template {
    
	static int mod = 1000000007;

	public static void main(String[] args){
        FastScanner sc = new FastScanner();
        PrintWriter out = new PrintWriter(System.out);
        int N = 100001;
        int yo = sc.nextInt();
        while (yo-- > 0) {
            List<List<Integer>> list = new ArrayList<>();
            int n = sc.nextInt();
            for(int i = 0; i < n; i++){
                list.add(new ArrayList<>());
            }
            
            List<Pair> res = new ArrayList<>();
            for(int i = 0; i < n-1; i++){
                int u = sc.nextInt()-1;
                int v = sc.nextInt()-1;
                res.add(new Pair(u+1,v+1));
                list.get(v).add(u);
                list.get(u).add(v);
            }
            
            boolean ok = helper(0,list,n,-1);
            if(ok){
                out.println(-1);
                continue;
            }
            map.clear();
            dfs(0,list,n,-1,-1);
            for(Pair p : res){
                int x = p.x;
                int y = p.y;
                out.print(map.get(x + " " + y) + " " );
            }
            out.println();
            
        }
        out.close();
	}
    
    static Map<String,Integer> map = new HashMap<>();
    static void dfs(int curr, List<List<Integer>> list, int n, int par, int what){
        List<Integer> neighbours = list.get(curr);
        if(what == -1){
            boolean three = true;
            for(int e : neighbours){
                String str1 = (curr+1) + " " + (e+1);
                String str2 = (e+1) + " " + (curr+1);
                // out.println(str1);
                if(three){
                    map.put(str1,3); map.put(str2,3);
                    three = false;
                }
                else {
                    map.put(str1,2);map.put(str2,2);
                }
                dfs(e,list,n,curr,map.get(str1));
            }
        }
        else {
            for(int e : neighbours){
                if(e == par) continue;
                String str1 = (curr+1) + " " + (e+1);
                String str2 = (e+1) + " " + (curr+1);
                if(what == 2){
                    map.put(str1,3);map.put(str2,3);
                }
                else {
                    map.put(str1,2); map.put(str2,2);
                }
                dfs(e,list,n,curr,map.get(str1));
            }    
        }
        
    }
    
    static boolean helper(int curr, List<List<Integer>> list, int n, int par){
        if(par != -1){
            if(list.get(curr).size() >= 3){
                return true;
            }
        }
        else {
            if(list.get(curr).size() > 2){
                return true;
            }
        }
        List<Integer> neighbours = list.get(curr);
        for(int e : neighbours){
            if(e == par) continue;
            if(helper(e,list,n,curr)){
                return true;
            }
        }
        return false;
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

	public static class Pair {
		int x;
		int y;

		public Pair(int x, int y) {
			this.x = x;
			this.y = y;
		}
	}

	

	

	

	

	public static void print(int[] arr, PrintWriter out) {
		//for debugging only
		for (int x : arr)
			out.print(x + " ");
		out.println();
	}
    
    
    
    

	

	//	For Input.txt and Output.txt	
	//	FileInputStream in = new FileInputStream("input.txt");
	//	FileOutputStream out = new FileOutputStream("output.txt");
	//	PrintWriter pw = new PrintWriter(out);
	//	Scanner sc = new Scanner(in);
}
