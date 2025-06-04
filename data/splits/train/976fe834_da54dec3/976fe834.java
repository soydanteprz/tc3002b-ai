import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.abs;
import static java.lang.System.out;
import java.util.*;
import java.io.*;
import java.math.*;
/*
-> Give your 100%, that's it!
-> Feynman Rule To Solve Any Problem:
    1. Read the problem.
    2. Think About It.
    3. Solve it!
*/

public class Template {
    
	static int mod = 1000000007;

	public static void main(String[] args){
        FastScanner sc = new FastScanner();
        PrintWriter out = new PrintWriter(System.out);
        int yo = sc.nextInt();
        while (yo-- > 0) {
            int n = sc.nextInt();
            int m = sc.nextInt();
            int[][] arr = new int[n][m];
            for(int i = 0; i < n; i++){
                String s = sc.next();
                for(int j = 0; j < m; j++){
                    arr[i][j] = s.charAt(j)-'0';
                }
            }
            
            List<String> list = new ArrayList<>();
            for(int i = 0; i < n-1; i++){
                for(int j = 0; j < m-1; j++){
                    check(list,i,j,n,m,arr);
                }
            }
            
            // for(int i = 0; i < n; i++){
            //     for(int j = 0; j < m; j++){
            //         out.print(arr[i][j] + " ");
            //     }
            //     out.println();
            // }
            
            out.println(list.size()/3);
            for(int i = 0; i < list.size(); i+=3){
                out.println(list.get(i) + " " + list.get(i+1) + " " + list.get(i+2));
            }
            // out.println(list.size());
            // for(int i = 0; i < list.size(); i++){
            //     out.println(list.get(i));
            // }
        }
        out.close();    
	}
    
    static void check(List<String> list, int i, int j, int n, int m, int[][] arr){
        int[][] a = new int[2][2];
        a[0][0] = arr[i][j];
        a[0][1] = arr[i][j+1];
        a[1][0] = arr[i+1][j];
        a[1][1] = arr[i+1][j+1];
        
        int ones = 0;
        for(int x = 0; x < 2; x++){
            for(int y = 0; y < 2; y++){
                if(a[x][y] == 1) ones++;
            }
        }
        
        if(ones == 0) return;
        
        if(ones == 1){
            ones(i,j,list,a);
            twos(i,j,list,a);
            threes(i,j,list,a);
        }
        else if(ones == 2){
            twos(i,j,list,a);
            threes(i,j,list,a);
        }
        else if(ones == 3){
            threes(i,j,list,a);
        }
        else if(ones == 4){
            fours(i,j,list,a);
            // for(int x = 0; x < 2; x++){
            //     for(int y = 0; y < 2; y++){
            //         out.print(a[x][y] + " ");
            //     }
            //     out.println();
            // }
            ones(i,j,list,a);
            // for(int x = 0; x < 2; x++){
            //     for(int y = 0; y < 2; y++){
            //         out.print(a[x][y] + " ");
            //     }
            //     out.println();
            // }
            twos(i,j,list,a);
            // for(int x = 0; x < 2; x++){
            //     for(int y = 0; y < 2; y++){
            //         out.print(a[x][y] + " ");
            //     }
            //     out.println();
            // }
            threes(i,j,list,a);
            // for(int x = 0; x < 2; x++){
            //     for(int y = 0; y < 2; y++){
            //         out.print(a[x][y] + " ");
            //     }
            //     out.println();
            // }
        }
        
        arr[i][j] = a[0][0];
        arr[i+1][j] = a[1][0];
        arr[i][j+1] = a[0][1];
        arr[i+1][j+1] = a[1][1];
    }
    
    public static void twos(int i, int j, List<String> list, int[][] a){
        int count = 0;
        for(int x = 0; x < 2; x++){
            for(int y = 0; y < 2; y++){
                if(a[x][y] == 1 && count < 1) {
                    count++;
                    a[x][y] = 0;
                    list.add(get(x,y,i,j));
                }
                else if(a[x][y] == 0){
                    list.add(get(x,y,i,j));
                    a[x][y] = 1;
                }
            }
        }
    }
    
    public static void threes(int i, int j, List<String> list, int[][] a){
        int count = 0;
        for(int x = 0; x < 2; x++){
            for(int y = 0; y < 2; y++){
                if(a[x][y] == 1) {
                    a[x][y] = 0;
                    list.add(get(x,y,i,j));
                }
            }
        }
    }
    
    public static void fours(int i, int j, List<String> list, int[][] a){
        int count = 0;
        for(int x = 0; x < 2; x++){
            for(int y = 0; y < 2; y++){
                if(a[x][y] == 1 && count < 3) {
                    count++;
                    a[x][y] = 0;
                    list.add(get(x,y,i,j));
                }
            }
        }
    }
    
    public static void ones(int i, int j, List<String> list, int[][] a){
        int count = 0;
        for(int x = 0; x < 2; x++){
            for(int y = 0; y < 2; y++){
                if(a[x][y] == 1) {
                    a[x][y] = 0;
                    list.add(get(x,y,i,j));
                }
                else if(a[x][y] == 0 && count < 2){
                    a[x][y] = 1;
                    count++;
                    list.add(get(x,y,i,j));
                }
            }
        }
    }
    
    public static String get(int x, int y, int i, int j){
        if(x == 0 && y == 0) {
            return (i+1) + " " + (j+1);
        }
        else if(x == 0 && y == 1){
            return (i+1) + " " + (j+2);
        }
        else if(x == 1 && y == 0){
            return (i+2) + " " + (j+1);
        }
        else {
            return (i+2) + " " + (j+2);
        }
    }
    
   
    
    
    /*
         Random stuff to try when stuck:
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
