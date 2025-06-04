import java.io.*;
import java.lang.reflect.Array;
import java.math.BigInteger;
import java.util.*;
import java.util.BitSet;
import java.util.function.BinaryOperator;

public class D {
    

    static FastReader s = new FastReader();
    static PrintWriter out = new PrintWriter(System.out);

    private static int[] rai(int n) {
        int[] arr = new int[n];
        for (int i = 0; i < n; i++) {
            arr[i] = s.nextInt();
        }
        return arr;
    }

    private static int[][] rai(int n, int m) {
        int[][] arr = new int[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                arr[i][j] = s.nextInt();
            }
        }
        return arr;
    }

    

    

    private static int ri() {
        return s.nextInt();
    }

    

    


    

    static int MOD=(int)1e9+7;

    

    
    public static void main(String[] args) {
        StringBuilder ans = new StringBuilder();
        int t = ri();
//        int t=1;
        while(t-- >0)
        {
            int n=ri();
            int[] arr=rai(n);
            int[] res = new int[n];
            if(n%2==0)
            {

                for(int i=0;i<n;i+=2)
                {
                    res[i] = -arr[i+1];
                    res[i+1]=arr[i];
                }


            }
            else {
                for(int i=0;i<n-3;i+=2)
                {
                    res[i] = -arr[i+1];
                    res[i+1]=arr[i];
                }

                int val = arr[n-1]+arr[n-2];
                if(val!=0) {
                    res[n - 3] = -val;
                    res[n - 2] = arr[n - 3];
                    res[n - 1] = arr[n - 3];
                }
                else
                {
                    val = arr[n-1]+arr[n-3];
                    if(val!=0) {
                        res[n - 2] = -val;
                        res[n - 1] = arr[n - 2];
                        res[n - 3] = arr[n - 2];
                    }
                    else
                    {
                        val = arr[n-2]+arr[n-3];
                        res[n-1] = -val;
                        res[n-2]=arr[n-1];
                        res[n-3]=arr[n-1];
                    }
                }

            }

            for(int i:res)
            {
                ans.append(i).append(" ");
            }
            ans.append("\n");

//            int sum=0;
//            for(int i=0;i<n;i++)
//            {
//                sum+=arr[i]*res[i];
//            }
//            System.out.println(sum);

        }
        out.print(ans.toString());
        out.flush();
    }
}
