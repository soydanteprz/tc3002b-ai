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
            int n, m, x;
            n = in.nextInt();
            m = in.nextInt();
            x = in.nextInt();

            int[] arr = new int[n];
            answer[] array = new answer[n];
            int[] ar = new int[n];

            for (int i = 0; i < n; i++) {
                arr[i] = in.nextInt();
                array[i] = new answer(arr[i],i);
            }

            long[] ans = new long[m];
            Arrays.sort(array);


            int a = 0 , b = 0;
            while(true){
                for (int i = 0; i < m; i++) {
                    ar[b] = i+1;
                    b++;
                    if(b==n){
                        break;
                    }
                }
                if(b==n){
                    break;
                }
                for (int i = m-1; i >= 0; i--) {
                    ar[b] = i+1;
                    b++;
                    if(b==n){
                        break;
                    }
                }

                if(b==n){
                    break;
                }
            }

            for (int i = 0; i < n; i++) {
                ans[ar[i]-1] += array[i].a;
            }

            for (int i = 0; i < m-1; i++) {
                if(Math.abs(ans[i]- ans[i+1])>x){
                    out.println("NO");
                    return;
                }
            }
            out.println("YES");

            int[] answer = new int[n];

            for (int i = 0; i < n; i++) {
                answer[array[i].b] = ar[i];
            }

            for (int i = 0; i < n; i++) {
                out.print(answer[i]+" ");
            }
            out.println();

        }
    }

    

    

    static class answer implements Comparable<answer>{
        int a;
        int b;

        public answer(int a, int b) {
            this.a = a;
            this.b = b;
        }

        

        
    }

    

    

    static void sort(long[] a) {
        ArrayList<Long> l=new ArrayList<>();
        for (Long i:a) l.add(i);
        l.sort(Collections.reverseOrder());
        for (int i=0; i<a.length; i++) a[i]=l.get(i);
    }

    static final Random random=new Random();

    

    
}
