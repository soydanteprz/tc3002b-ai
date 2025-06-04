import java.util.*;
import java.io.*;

public class q3{
    static FastScanner fs = new FastScanner();
    static PrintWriter pw = new PrintWriter(System.out);

    static int[] sec;
    static int[] health;
    static int[] pp;
    public static void main(String[] args){
        int T = fs.nextInt();
        for (int tt=0;tt<T;tt++){
            int n = fs.nextInt();
            sec = new int[n];
            health = new int[n];

            for (int i=0;i<n;i++) sec[i] = fs.nextInt();
            for (int i=0;i<n;i++) health[i] = fs.nextInt();

            pp = new int[]{sec[n-1]-health[n-1]+1, sec[n-1], health[n-1]};

            long sum = 0;
            for (int i=n-2;i>=0;i--){
                int left = sec[i] - health[i] + 1, right = sec[i], top = health[i];

                if (right < pp[0]){
                    sum += ((long)pp[2] + 1) * (long)pp[2] / 2;
                    pp = new int[]{left, right, top};
                }
                else if (left < pp[0]){
                    pp[0] = left;
                    pp[2] = top + (pp[1] - right);
                }
            }
            sum += ((long)pp[2] + 1) * (long)pp[2] / 2;

            pw.println(sum);
        }



        pw.close();
    }

//    ----------input function----------

    

    
}
