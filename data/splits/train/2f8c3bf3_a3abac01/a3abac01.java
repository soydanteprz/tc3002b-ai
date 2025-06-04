import  java.math.*;
import java.util.*;
import java.lang.*;
import java.io.*;



public final class Main {

    FastReader s;


    public static void main (String[] args) throws java.lang.Exception
    {

        new Main().run();

    }
    void run()
    {
        s = new FastReader();
        solve();
    }
   StringBuffer sb;
    void solve()
    {
         sb = new StringBuffer();
        for(int T = s.nextInt();T > 0;T--)
            start();
       // System.out.print(sb);

    }


    void start()
    {
        int n = s.nextInt();
        int m = s.nextInt();
        long arr[] = longArr(m);
        long arr2[] = longArr(m);
        long ans [] = new long[n];
        Arrays.fill(ans,Integer.MAX_VALUE);
        for(int i = 0; i<m; i++)
        {
            ans[(int)arr[i]-1] = arr2[i];
        }
        long left [] = new long[n];
        long right [] = new long[n];
        left[0] = ans[0];
        for(int i = 1; i<n; i++)
        {
            left[i] = Math.min(ans[i], left[i-1]+1);
        }
        right[n-1] = ans[n-1];
        for(int i = n-2; i>=0; i--)
        {
            right[i] = Math.min(right[i+1]+1, ans[i]);
        }
        StringBuffer x = new StringBuffer();
        for(int i =0; i<n; i++)
        {
            long p = Math.min(left[i],right[i]);
            x.append(p+" ");
        }
        System.out.println(x);

    }
    
    
    

    

    


    
    
    

    //long array input
    public long [] longArr(int len)
    {
        // long arr input
        long [] arr = new long[len];
        String [] strs =  s.nextLine().trim().split("\\s+");
        for(int i =0; i<len; i++)
        {
            arr[i] = Long.parseLong(strs[i]);

        }
        return arr;
    }
    // int arr input
    

    

}
