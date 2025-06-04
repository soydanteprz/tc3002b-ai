import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigInteger;

import java.util.*;

public class Main
{

  

    public static void main(String[] args)
    {
        FastScanner input = new FastScanner();
       StringBuilder result = new StringBuilder();
        int tc = input.nextInt();
        while (tc-- > 0) {
          int n = input.nextInt();
      int a[] = new int[n];
      int b[] = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = input.nextInt();
            
        }
        if(n%2==0)
        {
            for (int i = 0; i <n; i+=2) {
                b[i] = a[i+1];
                b[i+1] = -a[i];
            }
        }
        else
        {
            
            for (int i = 0; i <n-3; i+=2) {
                b[i] = a[i+1];
                b[i+1] = -a[i];
            }
            int last1 = a[n-3];
            int last2 = a[n-2];
            int last3 = a[n-1];
            if(last1+last2!=0)
            {
                b[n-3] = -last3;
                b[n-2] = -last3;
                b[n-1] = (last1+last2);
            }
            else if(last1+last3!=0)
            {
                b[n-3] = -last2;
                b[n-2] = (last1+last3);
                b[n-1] = -last2;
            }
            else if(last2+last3!=0)
            {
                b[n-3] =(last2+last3);
                b[n-2] = -last1;
                b[n-1] = -last1;
            }
        }
        for (int i : b) {
            result.append(i+" ");
        }
        result.append("\n");   
        }
        System.out.println(result);
        
      
    }

    

}
