import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.*;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class First {

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
        public void call(InputReader in, PrintWriter out) {
            int n;
            n = in.nextInt();
            int[] arr = new int[n];
            long[] array = new long[n];

            for (int i = 0; i < n; i++) {
                arr[i] = in.nextInt();
                if(i==0){
                    array[i] = arr[i];
                }
                else
                    array[i] = (long)arr[i] + array[i-1];
            }

            int evenMin = arr[0], oddMin = arr[1];

            long ans = ((long) n * (long)arr[0]) + ((long) n * (long)arr[1]);

            for (int i = 2; i < n; i++) {
                if(i%2==0){
                    if(evenMin>arr[i]) {
                        evenMin = arr[i];
                    }
                }
                else{
                    if(oddMin>arr[i]) {
                        oddMin = arr[i];
                    }
                }
                if(i%2==0){
                    ans = Math.min(ans, array[i] + (long) ((n - (i / 2)) - 1) * (long)evenMin + (long) ((n - ((i-1) / 2)) - 1) * (long)oddMin );
                }
                else{
                    ans = Math.min(ans, array[i] + (long) ((n - (i / 2)) - 1) * (long)oddMin + (long) ((n - ((i-1) / 2)) - 1) * (long)evenMin);
                }

            }
            out.println(ans);





        }
    }

    

    

    

    

    

    

    static final Random random=new Random();

    

    
}
