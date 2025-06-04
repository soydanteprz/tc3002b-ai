import java.util.*;
import java.io.*;
import java.lang.*;
import java.util.StringTokenizer;
import java.util.concurrent.TimeUnit;

public class C101 {

    static HritikScanner sc = new HritikScanner();
    static PrintWriter pw = new PrintWriter(System.out, true);

    public static void main(String[] args) {

        int t = ni();
        while (t-- > 0) {
            solve();
        }
    }

     static void solve() {
         int n = ni();
         int m = ni(), x = ni();
         Map<Integer, ArrayList<Integer>> map = new TreeMap<>();
         System.out.println("YES");
         for(int i = 0; i < n; i++)
         {
             int num = ni();
             if(!map.containsKey(num))
             {
                 map.put(num, new ArrayList<Integer>());
             }
             map.get(num).add(i);
         }
         int[] ans = new int[n];
         int q = 0;
         for(int block : map.keySet())
         {
             for(int i = 0; i < map.get(block).size(); i++)
             {
                 ans[map.get(block).get(i)] = (q++)%m+1;
             }
         }
         for(int ele : ans)
         {
             System.out.print(ele + " ");
         }
     }
    
    
    /////////////////////////////////////////////////////////////////////////////////
    

    

    

    /////////////////////////////////////////////////////////////////////////////////
    static int ni() {
        return sc.nextInt();
    }

    

    

    /////////////////////////////////////////////////////////////////////////////////

    

    

    

    

    

    

    

    

    

    

    

    

/////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////
    //-----------HritikScanner class for faster input----------//
    

    //////////////////////////////////////////////////////////////////
    

    //////////////////////////////////////////////////////////////////
    // Function to return gcd of a and b time complexity O(log(a+b))
    

    // method to return LCM of two numbers 
    

    //////////////////////////////////////////////////////////////////
    
    //////////////////////////////////////////////////////////////////

    

    

    

    

    //////////////////////////////////////////////////////////////////
    // method returns Nth power of A 
    

}
