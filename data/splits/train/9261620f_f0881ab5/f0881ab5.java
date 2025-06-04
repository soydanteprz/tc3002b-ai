import java.util.*;
    public class Main {
        public static void main(String args[]) {
            Scanner sc = new Scanner(System.in);
            int t = sc.nextInt();
            while(t-->0){
                int n  =sc.nextInt();
                long[] arr = new long[n];
                for(int i=0;i<n;i++){
                    arr[i] =sc.nextLong();
                }
                Map<Long, Long> map = new HashMap<>();
                List<Long> list = new ArrayList<>();
                list.add((long)0);
                for(int i=0;i<n;i++){
                    int s = list.size();
                    for(int j=0;j<s;j++){
                        list.add(list.get(j)+arr[i]);
                    }
                }
                int flag = 0;
                for(int i=0;i<list.size();i++){
                    map.put(list.get(i), map.getOrDefault(list.get(i), (long)0)+1);
                    if(map.get(list.get(i))>1){
                        flag = 1;
                        break;
                    }
                }
                if(flag>0)
                    System.out.println("YES");
                else
                    System.out.println("NO");
            }
        }
    }
