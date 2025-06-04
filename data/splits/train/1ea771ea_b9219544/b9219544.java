import static java.lang.Math.*;
import static java.lang.System.out;
import java.util.*;
import java.io.*;
import java.math.*;

public class x1547E
{
    public static void main(String hi[]) throws Exception
    {
        FastScanner infile = new FastScanner();
        StringBuilder sb = new StringBuilder();
        int T = infile.nextInt();
        while(T-->0)
        {
            int N = infile.nextInt();
            int K = infile.nextInt();
            int[] locs = infile.nextInts(K);
            int[] temps = infile.nextInts(K);
            long[] arr = new long[N];
            Arrays.fill(arr, Long.MAX_VALUE/2);
            for(int i=0; i < K; i++)
                arr[locs[i]-1] = temps[i];
            long[] suffix = new long[N];
            suffix[N-1] = arr[N-1];
            for(int i=N-2; i >= 0; i--)
                suffix[i] = min(arr[i], suffix[i+1]+1);
            long[] prefix = new long[N];
            prefix[0] = arr[0];
            for(int i=1; i < N; i++)
                prefix[i] = min(arr[i], prefix[i-1]+1);
            for(int i=0; i < N; i++)
            {
                long res = min(prefix[i], suffix[i]);
                sb.append(res+" ");
            }
            sb.append("\n");
        }
        System.out.print(sb);
    }
}

