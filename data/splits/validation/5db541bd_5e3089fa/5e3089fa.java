import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SetOrDecrease {
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int t = Integer.parseInt(br.readLine());
        while(t>0){
            String []nk=br.readLine().split(" ");
            int n=Integer.parseInt(nk[0]);
            long k=Long.parseLong(nk[1]);
            String []str=br.readLine().split(" ");
            List<Integer> list=new ArrayList<>();
            for(int i=0;i<n;i++)
                list.add(Integer.parseInt(str[i]));
            Collections.sort(list);
            long sum=0;
            for(int i=0;i<n;i++)
                sum+=list.get(i);
            if(sum<=k)
                System.out.println(0);
            else {
                long minSteps = sum - k;
                long drop=0;
                long targetDrop = sum-k;
                for (int i = n - 1; i > 0; i--) {
                    //assignment at ith element
                    drop += list.get(i) - list.get(0);
                    if (drop >= targetDrop) {
                        if (minSteps > n - i)
                            minSteps = n - i;
                    } else {
                        long diff = targetDrop - drop;
                        long div = diff / (n - i + 1);
                        long rem = diff % (n - i + 1);
                        long steps = rem > 0 ? (div + 1) + n - i : div + n - i;
                        if (minSteps > steps)
                            minSteps = steps;

                    }
                }
                System.out.println(minSteps);
            }
            t--;
        }
    }
}
