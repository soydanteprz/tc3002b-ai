import java.util.*;
import java.io.*;
import java.lang.*;
import java.util.StringTokenizer;

public class Solution {

    static HritikScanner sc = new HritikScanner();
    static PrintWriter pw = new PrintWriter(System.out, true);
    static StringBuilder sb = new StringBuilder();

    public static void main(String[] args) {

        int t = ni();
        while (t-- > 0) {
            solve();
        }
    }

    static void solve() {
        int n = ni();
        int[] arr = nextIntArray(n);
        char[] col = sc.next().toCharArray();
        int[] cB = new int[n+1];
        int[] cR = new int[n+1];
        for(int i = 0; i < n; i++)
        {
//            pl((col[i] == 'R' && arr[i] > n));
//            pl((col[i] == 'B' && arr[i] < 1));
            if((col[i] == 'R' && arr[i] > n)||(col[i] == 'B' && arr[i] < 1))
            {
                System.out.println("NO");
                return;
            }
            if(col[i] == 'B')
            {
                if(arr[i] > n)
                    continue;
                cB[arr[i]]++;
                if(cB[arr[i]] > arr[i])
                {
                    System.out.println("NO");
                    return;
                }
            }
            else
            {
                if(arr[i] < 1)
                    continue;
                cR[arr[i]]++;
                if(cR[arr[i]] > (n-(arr[i]-1)))
                {
                    System.out.println("NO");
                    return;
                }
            }
//            pa(cB);
//            pa(cR);
        }
//        pa(cB);
//        pa(cR);
        int[] psum = new int[n+1];
        for(int i = 1; i<= n; i++)
        {
            psum[i] = psum[i-1]+cB[i];
            if(psum[i] > i)
            {
                System.out.println("NO");
                return;
            }
        }
//        pa(psum);
        int[] psum1 = new int[n+1];
        psum1[n] = cR[n];
        for(int i = n-1; i>= 0; i--)
        {
            psum1[i] = psum1[i+1]+cR[i];
            if(psum1[i] > (n-(i-1)))
            {
                System.out.println("NO");
                return;
            }
        }
//        pa(psum1);
        
        System.out.println("YES");
        
    }
        /////////////////////////////////////////////////////////////////////////////////
    

    /////////////////////////////////////////////////////////////////////////////////
    static int[] nextIntArray(int n) {
        int[] arr = new int[n];
        int i = 0;
        while (i < n) {
            arr[i++] = ni();
        }
        return arr;
    }

    

    

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
