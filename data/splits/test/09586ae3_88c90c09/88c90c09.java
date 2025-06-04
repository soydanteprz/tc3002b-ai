import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.sql.SQLOutput;
import java.util.*;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        TaskA solver = new TaskA();
        //int a = 1;
        int t;
        t = in.nextInt();
        //t = 1;
        while (t > 0) {
            //out.print("Case #"+(a++)+": ");
            solver.call(in,out);
            t--;
        }
        out.close();

    }

    static class TaskA {
        int n;
        long[] arr;
        public void call(InputReader in, PrintWriter out) {
            n = in.nextInt();
            arr = new long[n];

            for (int i = 0; i < n; i++) {
                arr[i] = in.nextLong();
            }
            long l, r, mid;
            l = 0;
            r = (long)1e10;
            while (l  + 1 < r){
                mid = ( l + r)/2;
                if(ans(mid)){
                    l = mid;
                }
                else{
                    r = mid;
                }
            }

            out.println(l);
        }

        public boolean ans(long mid) {
            long[] array = new long[n];
            long a, b;
            for (int i = n - 1; i >= 2; i--) {
                a = mid - array[i];
                if(a > 0){
                    b = arr[i] - a;
                    if(b >= 0){
                        b/=3;
                        array[i - 1] += b;
                        array[i - 2] += 2*b;
                    }
                    else{
                        return false;
                    }
                }
                else{
                    b = arr[i];
                    b/=3;
                    array[i - 1] += b;
                    array[i - 2] += 2*b;
                }
            }
            for (int i = 0; i < 2; i++) {
                if(arr[i] + array[i] < mid){
                    return false;
                }
            }
            return true;
        }
    }

    

    


    

    

    

    

    static final Random random=new Random();

    

    
}
