import java.awt.Container;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.math.BigInteger;

import java.util.*;

public class Main
{
   
    public static boolean check(int a[],int value)
    {
           int b[]= new int[a.length];
           for (int i = a.length-1; i >=2; i--) {
            if(a[i]+b[i]<value)
            {
                return false;
            }
            int d = Math.min(a[i], (a[i]+b[i]-value));
            b[i-1]+=d/3;
            b[i-2]+=(2*(d/3));
            
        }
        
        
        
        if(a[0]+b[0]<value||a[1]+b[1]<value)
        {
            return false;
        }
        return true;
        
    }
   
    public static void main(String[] args)
    {
        FastScanner input = new FastScanner();
        StringBuilder result = new StringBuilder();
        int tc = input.nextInt();
        work:
        while (tc-- > 0) {
          int n = input.nextInt();
        int a[]= new int[n];
     
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            a[i] = input.nextInt();
            min = Math.min(min, a[i]);
            max = Math.max(max, a[i]);
            
        }
        int low = min;
        int high  = max;
        int ans = 0;
        while(low<=high)
        {
            int mid = low+(high-low)/2;
            if(check(a, mid))
            {
                ans = mid;
                low = mid+1;
            }
            else
            {
                high = mid-1;
            }
            
            
        }
        result.append(ans+"\n");
           
        }
        System.out.println(result);
        
       
    }

    

}
